import argparse
import os
import enum
from typing import List, Optional, Tuple
import ast

import torch
import numpy as np
from PIL import Image
from PIL.Image import Resampling

import coremltools as ct
from coremltools.converters.mil._deployment_compatibility import AvailableTarget
from coremltools import ComputeUnit
from coremltools.converters.mil.mil.passes.defs.quantization import ComputePrecision
from coremltools.converters.mil import register_torch_op
from coremltools.converters.mil.mil import Builder as mb

from sam2.sam2_video_predictor import SAM2VideoPredictor
from sam2.sam2_image_predictor import SAM2ImagePredictor


class SAM2Variant(enum.Enum):
    Tiny = "tiny"
    Small = "small"
    BasePlus = "base-plus"
    Large = "large"

    def fmt(self):
        if self == SAM2Variant.BasePlus:
            return "BasePlus"
        return self.value.capitalize()


SAM2_HW = (1024, 1024)


def parse_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Provide location to save exported models.",
    )
    parser.add_argument(
        "--variant",
        type=lambda x: getattr(SAM2Variant, x),
        choices=[variant for variant in SAM2Variant],
        default=SAM2Variant.Small,
        help="SAM2 variant to export.",
    )
    parser.add_argument(
        "--points",
        type=str,
        help="List of 2D points, e.g., '[[10,20], [30,40]]'",
    )
    parser.add_argument(
        "--labels",
        type=str,
        help="List of binary labels for each points entry, denoting foreground (1) or background (0).",
    )
    parser.add_argument(
        "--min-deployment-target",
        type=lambda x: getattr(AvailableTarget, x),
        choices=[target for target in AvailableTarget],
        default=AvailableTarget.iOS17,
        help="Minimum deployment target for CoreML model.",
    )
    parser.add_argument(
        "--compute-units",
        type=lambda x: getattr(ComputeUnit, x),
        choices=[cu for cu in ComputeUnit],
        default=ComputeUnit.ALL,
        help="Which compute units to target for CoreML model.",
    )
    parser.add_argument(
        "--precision",
        type=lambda x: getattr(ComputePrecision, x),
        choices=[p for p in ComputePrecision],
        default=ComputePrecision.FLOAT16,
        help="Precision to use for quantization.",
    )
    return parser


@register_torch_op
def upsample_bicubic2d(context, node):
    x = context[node.inputs[0]]
    output_size = context[node.inputs[1]].val

    scale_factor_height = output_size[0] / x.shape[2]
    scale_factor_width = output_size[1] / x.shape[3]

    align_corners = context[node.inputs[2]].val
    x = mb.upsample_bilinear(
        x=x,
        scale_factor_height=scale_factor_height,
        scale_factor_width=scale_factor_width,
        align_corners=align_corners,
        name=node.name,
    )
    context.add(x)


class SAM2VideoImageEncoder(torch.nn.Module):
    def __init__(self, predictor: SAM2ImagePredictor):
        super().__init__()
        self.predictor = predictor

    @torch.no_grad()
    def forward(self, image):
        backbone_out = self.predictor.model.forward_image(image)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self.predictor.model._prepare_backbone_features(backbone_out)

        if self.predictor.model.directly_add_no_mem_embed:
            vision_feats[-1] = vision_feats[-1] + self.predictor.model.no_mem_embed

        feats = [
            feat.permute(1, 2, 0).view(image.shape[0], -1, *feat_size)
            for feat, feat_size in zip(
                vision_feats[::-1], feat_sizes[::-1]
            )
        ][::-1]

        image_embed = feats[-1]
        high_res_feats = feats[:-1]
        feats_s0, feats_s1 = high_res_feats[0], high_res_feats[1]

        image_pos = vision_pos_embeds[-1].permute(1, 2, 0).view(
            image.shape[0], -1, *feat_sizes[-1]
        )
        return image_embed, feats_s0, feats_s1, image_pos


class SAM2PointsEncoder(torch.nn.Module):
    def __init__(self, predictor: SAM2ImagePredictor):
        super().__init__()
        self.predictor = predictor

    @torch.no_grad()
    def forward(self, points, labels):
        prompt_embedding = self.predictor.encode_points_raw(points, labels)
        return prompt_embedding


class SAM2MaskDecoder(torch.nn.Module):
    def __init__(self, predictor: SAM2ImagePredictor):
        super().__init__()
        self.predictor = predictor

    @torch.no_grad()
    def forward(
        self, image_embedding, sparse_embedding, dense_embedding, feats_s0, feats_s1
    ):
        low_res_masks, iou_scores = self.predictor.decode_masks_raw(
            image_embedding,
            sparse_embedding,
            dense_embedding,
            [feats_s0, feats_s1],
        )
        return low_res_masks, iou_scores


class SAM2VideoMemoryEncoder(torch.nn.Module):
    def __init__(self, predictor: SAM2VideoPredictor):
        super().__init__()
        self.predictor = predictor

    @torch.no_grad()
    def forward(self, pix_feat, masks):
        out = self.predictor.memory_encoder(
            pix_feat,
            masks,
            skip_mask_sigmoid=True,
        )
        features = out["vision_features"]
        pos_enc = out["vision_pos_enc"][0]
        return features, pos_enc


class SAM2VideoMemoryAttention(torch.nn.Module):
    def __init__(self, predictor: SAM2VideoPredictor):
        super().__init__()
        self.predictor = predictor

    @torch.no_grad()
    def forward(
        self,
        current_feat: torch.Tensor,
        current_pos: torch.Tensor,
        memory: torch.Tensor,
        memory_pos: torch.Tensor,
        num_obj_ptr_tokens: torch.Tensor,
    ):
        batch_size, channels, height, width = current_feat.shape
        seq = current_feat.flatten(2).permute(2, 0, 1)
        seq_pos = current_pos.flatten(2).permute(2, 0, 1)
        ptr_tokens = int(num_obj_ptr_tokens.item())
        fused = self.predictor.memory_attention(
            curr=seq,
            curr_pos=seq_pos,
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=ptr_tokens,
        )
        fused = fused.permute(1, 2, 0).view(batch_size, channels, height, width)
        return fused


# Validation helpers

def validate_image_encoder(
    model: ct.models.MLModel,
    predictor: SAM2ImagePredictor,
    image: Image.Image,
):
    prepared = image.resize(SAM2_HW, Resampling.BILINEAR)
    predictions = model.predict({"image": prepared})

    image_np = np.array(image.convert("RGB"))
    torch_image = predictor._transforms(image_np)
    torch_image = torch_image[None, ...].to("cpu")

    (image_embed, feats_s0, feats_s1) = predictor.encode_image_raw(torch_image)
    backbone_out = predictor.model.forward_image(torch_image)
    (
        _,
        _,
        vision_pos_embeds,
        feat_sizes,
    ) = predictor.model._prepare_backbone_features(backbone_out)
    image_pos = vision_pos_embeds[-1].permute(1, 2, 0).view(
        torch_image.shape[0], -1, *feat_sizes[-1]
    )

    def stats(coreml_out, torch_out, name):
        max_diff = np.max(np.abs(coreml_out - torch_out))
        avg_diff = np.mean(np.abs(coreml_out - torch_out))
        print(f"{name}: Max Diff: {max_diff:.4f}, Avg Diff: {avg_diff:.4f}")

    stats(predictions["image_embedding"], image_embed.numpy(), "Image Embedding")
    stats(predictions["feats_s0"], feats_s0.numpy(), "Feats S0")
    stats(predictions["feats_s1"], feats_s1.numpy(), "Feats S1")
    stats(predictions["image_pos_enc"], image_pos.numpy(), "Image Pos Enc")


def validate_prompt_encoder(
    model: ct.models.MLModel,
    predictor: SAM2ImagePredictor,
    unnorm_coords,
    labels,
):
    predictions = model.predict({"points": unnorm_coords, "labels": labels})

    (ground_sparse, ground_dense) = predictor.encode_points_raw(
        torch.from_numpy(unnorm_coords).float(), torch.from_numpy(labels).int()
    )
    ground_sparse = ground_sparse.numpy()
    ground_dense = ground_dense.numpy()

    sparse_max_diff = np.max(
        np.abs(predictions["sparse_embeddings"] - ground_sparse)
    )
    sparse_avg_diff = np.mean(
        np.abs(predictions["sparse_embeddings"] - ground_sparse)
    )

    dense_max_diff = np.max(
        np.abs(predictions["dense_embeddings"] - ground_dense)
    )
    dense_avg_diff = np.mean(
        np.abs(predictions["dense_embeddings"] - ground_dense)
    )

    print(
        "Sparse Embeddings: Max Diff: {:.4f}, Avg Diff: {:.4f}".format(
            sparse_max_diff, sparse_avg_diff
        )
    )
    print(
        "Dense Embeddings: Max Diff: {:.4f}, Avg Diff: {:.4f}".format(
            dense_max_diff, dense_avg_diff
        )
    )


def validate_mask_decoder(
    model: ct.models.MLModel,
    predictor: SAM2ImagePredictor,
    image_embedding,
    sparse_embedding,
    dense_embedding,
    feats_s0,
    feats_s1,
    precision: ComputePrecision,
):
    predictions = model.predict(
        {
            "image_embedding": image_embedding,
            "sparse_embedding": sparse_embedding,
            "dense_embedding": dense_embedding,
            "feats_s0": feats_s0,
            "feats_s1": feats_s1,
        }
    )

    ground_masks, scores = predictor.decode_masks_raw(
        torch.from_numpy(image_embedding).float(),
        torch.from_numpy(sparse_embedding).float(),
        torch.from_numpy(dense_embedding).float(),
        [
            torch.from_numpy(feats_s0).float(),
            torch.from_numpy(feats_s1).float(),
        ],
    )

    ground_masks = ground_masks.numpy()
    masks_max_diff = np.max(np.abs(predictions["low_res_masks"] - ground_masks))
    masks_avg_diff = np.mean(np.abs(predictions["low_res_masks"] - ground_masks))

    print(
        "Masks: Max Diff: {:.4f}, Avg Diff: {:.4f}".format(
            masks_max_diff, masks_avg_diff
        )
    )

    print(f"Scores: {predictions['scores']}, ground: {scores}")
    assert np.allclose(predictions["scores"], scores.numpy(), atol=1e-2)


def validate_memory_encoder(
    model: ct.models.MLModel,
    predictor: SAM2VideoPredictor,
    pix_feat: torch.Tensor,
    masks: torch.Tensor,
):
    predictions = model.predict(
        {
            "pix_feat": pix_feat.numpy(),
            "masks": masks.numpy(),
        }
    )

    with torch.no_grad():
        out = predictor.memory_encoder(pix_feat, masks, skip_mask_sigmoid=True)
        ground_features = out["vision_features"].numpy()
        ground_pos = out["vision_pos_enc"][0].numpy()

    feat_diff = np.max(np.abs(predictions["maskmem_features"] - ground_features))
    pos_diff = np.max(np.abs(predictions["maskmem_pos_enc"] - ground_pos))
    print(
        f"Memory Encoder - Features Max Diff: {feat_diff:.4f}, Pos Max Diff: {pos_diff:.4f}"
    )


def validate_memory_attention(
    model: ct.models.MLModel,
    predictor: SAM2VideoPredictor,
    current_feat: torch.Tensor,
    current_pos: torch.Tensor,
    memory: torch.Tensor,
    memory_pos: torch.Tensor,
    num_obj_ptr_tokens: int,
):
    predictions = model.predict(
        {
            "current_feat": current_feat.numpy(),
            "current_pos": current_pos.numpy(),
            "memory": memory.numpy(),
            "memory_pos": memory_pos.numpy(),
            "num_obj_ptr_tokens": np.array([num_obj_ptr_tokens], dtype=np.int32),
        }
    )

    with torch.no_grad():
        seq = current_feat.flatten(2).permute(2, 0, 1)
        seq_pos = current_pos.flatten(2).permute(2, 0, 1)
        fused = predictor.memory_attention(
            curr=seq,
            curr_pos=seq_pos,
            memory=memory,
            memory_pos=memory_pos,
            num_obj_ptr_tokens=num_obj_ptr_tokens,
        )
        fused = fused.permute(1, 2, 0).view_as(current_feat)
        fused_np = fused.numpy()

    max_diff = np.max(np.abs(predictions["fused_embedding"] - fused_np))
    avg_diff = np.mean(np.abs(predictions["fused_embedding"] - fused_np))
    print(
        f"Memory Attention - Max Diff: {max_diff:.4f}, Avg Diff: {avg_diff:.4f}"
    )


# Export helpers

def export_image_encoder(
    predictor: SAM2ImagePredictor,
    variant: SAM2Variant,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
) -> Tuple[int, int]:
    image = Image.open("../notebooks/images/truck.jpg")
    image_np = np.array(image.convert("RGB"))
    orig_hw = (image_np.shape[0], image_np.shape[1])

    prepared_image = predictor._transforms(image_np)
    prepared_image = prepared_image[None, ...].to("cpu")

    traced_model = torch.jit.trace(
        SAM2VideoImageEncoder(predictor).eval(), prepared_image
    )

    scale = 1 / (0.226 * 255.0)
    bias = [-0.485 / (0.229), -0.456 / (0.224), -0.406 / (0.225)]

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.ImageType(
                name="image",
                shape=(1, 3, SAM2_HW[0], SAM2_HW[1]),
                scale=scale,
                bias=bias,
            )
        ],
        outputs=[
            ct.TensorType(name="image_embedding"),
            ct.TensorType(name="feats_s0"),
            ct.TensorType(name="feats_s1"),
            ct.TensorType(name="image_pos_enc"),
        ],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
        compute_precision=precision,
    )

    validate_image_encoder(mlmodel, predictor, image)

    output_path = os.path.join(
        output_dir,
        f"SAM2{variant.fmt()}VideoImageEncoder{precision.value.upper()}",
    )
    mlmodel.save(output_path + ".mlpackage")
    return orig_hw


def export_points_prompt_encoder(
    predictor: SAM2ImagePredictor,
    variant: SAM2Variant,
    input_points: List[List[float]],
    input_labels: List[int],
    orig_hw: tuple,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
):
    predictor.model.sam_prompt_encoder.eval()

    points = torch.tensor(input_points, dtype=torch.float32)
    labels = torch.tensor(input_labels, dtype=torch.int32)

    unnorm_coords = predictor._transforms.transform_coords(
        points,
        normalize=True,
        orig_hw=orig_hw,
    )
    unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]

    traced_model = torch.jit.trace(
        SAM2PointsEncoder(predictor), (unnorm_coords, labels)
    )

    points_shape = ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=16), 2))
    labels_shape = ct.Shape(shape=(1, ct.RangeDim(lower_bound=1, upper_bound=16)))

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="points", shape=points_shape),
            ct.TensorType(name="labels", shape=labels_shape),
        ],
        outputs=[
            ct.TensorType(name="sparse_embeddings"),
            ct.TensorType(name="dense_embeddings"),
        ],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
        compute_precision=precision,
    )

    validate_prompt_encoder(
        mlmodel,
        predictor,
        unnorm_coords.numpy(),
        labels.numpy(),
    )

    output_path = os.path.join(
        output_dir,
        f"SAM2{variant.fmt()}VideoPromptEncoder{precision.value.upper()}",
    )
    mlmodel.save(output_path + ".mlpackage")


def export_mask_decoder(
    predictor: SAM2ImagePredictor,
    variant: SAM2Variant,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
):
    predictor.model.sam_mask_decoder.eval()
    s0 = torch.randn(1, 32, 256, 256)
    s1 = torch.randn(1, 64, 128, 128)
    image_embedding = torch.randn(1, 256, 64, 64)
    sparse_embedding = torch.randn(1, 3, 256)
    dense_embedding = torch.randn(1, 256, 64, 64)

    traced_model = torch.jit.trace(
        SAM2MaskDecoder(predictor),
        (image_embedding, sparse_embedding, dense_embedding, s0, s1),
    )
    traced_model.eval()

    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(name="image_embedding", shape=[1, 256, 64, 64]),
            ct.TensorType(
                name="sparse_embedding",
                shape=ct.EnumeratedShapes(shapes=[[1, i, 256] for i in range(2, 16)]),
            ),
            ct.TensorType(name="dense_embedding", shape=[1, 256, 64, 64]),
            ct.TensorType(name="feats_s0", shape=[1, 32, 256, 256]),
            ct.TensorType(name="feats_s1", shape=[1, 64, 128, 128]),
        ],
        outputs=[
            ct.TensorType(name="low_res_masks"),
            ct.TensorType(name="scores"),
        ],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
        compute_precision=precision,
    )

    validate_mask_decoder(
        mlmodel,
        predictor,
        image_embedding.numpy(),
        sparse_embedding.numpy(),
        dense_embedding.numpy(),
        s0.numpy(),
        s1.numpy(),
        precision,
    )

    output_path = os.path.join(
        output_dir,
        f"SAM2{variant.fmt()}VideoMaskDecoder{precision.value.upper()}",
    )
    mlmodel.save(output_path + ".mlpackage")


def export_memory_encoder(
    video_model: SAM2VideoPredictor,
    variant: SAM2Variant,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
):
    batch = 2
    hidden_dim = video_model.hidden_dim
    mem_size = video_model.image_size // video_model.backbone_stride
    pix_feat = torch.randn(batch, hidden_dim, mem_size, mem_size)
    masks = torch.randn(batch, 1, SAM2_HW[0], SAM2_HW[1])

    traced_model = torch.jit.trace(
        SAM2VideoMemoryEncoder(video_model).eval(), (pix_feat, masks)
    )

    batch_shape = ct.RangeDim(lower_bound=1, upper_bound=8)
    mlmodel = ct.convert(
        traced_model,
        inputs=[
            ct.TensorType(
                name="pix_feat",
                shape=ct.Shape(
                    shape=(batch_shape, hidden_dim, mem_size, mem_size)
                ),
            ),
            ct.TensorType(
                name="masks",
                shape=ct.Shape(shape=(batch_shape, 1, SAM2_HW[0], SAM2_HW[1])),
            ),
        ],
        outputs=[
            ct.TensorType(name="maskmem_features"),
            ct.TensorType(name="maskmem_pos_enc"),
        ],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
        compute_precision=precision,
    )

    validate_memory_encoder(mlmodel, video_model, pix_feat, masks)

    output_path = os.path.join(
        output_dir,
        f"SAM2{variant.fmt()}MemoryEncoder{precision.value.upper()}",
    )
    mlmodel.save(output_path + ".mlpackage")


def export_memory_attention(
    video_model: SAM2VideoPredictor,
    variant: SAM2Variant,
    output_dir: str,
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
):
    batch = 2
    hidden_dim = video_model.hidden_dim
    mem_dim = video_model.mem_dim
    spatial = video_model.image_size // video_model.backbone_stride
    memory_tokens = video_model.num_maskmem + 8

    current_feat = torch.randn(batch, hidden_dim, spatial, spatial)
    current_pos = torch.randn(batch, hidden_dim, spatial, spatial)
    memory = torch.randn(memory_tokens, batch, mem_dim)
    memory_pos = torch.randn(memory_tokens, batch, mem_dim)
    num_obj_ptr_tokens = torch.tensor([4], dtype=torch.int32)

    scripted_model = torch.jit.script(
        SAM2VideoMemoryAttention(video_model).eval()
    )

    batch_shape = ct.RangeDim(lower_bound=1, upper_bound=8)
    memory_len = ct.RangeDim(lower_bound=1, upper_bound=64)

    mlmodel = ct.convert(
        scripted_model,
        inputs=[
            ct.TensorType(
                name="current_feat",
                shape=ct.Shape(
                    shape=(batch_shape, hidden_dim, spatial, spatial)
                ),
            ),
            ct.TensorType(
                name="current_pos",
                shape=ct.Shape(
                    shape=(batch_shape, hidden_dim, spatial, spatial)
                ),
            ),
            ct.TensorType(
                name="memory",
                shape=ct.Shape(
                    shape=(memory_len, batch_shape, mem_dim)
                ),
            ),
            ct.TensorType(
                name="memory_pos",
                shape=ct.Shape(
                    shape=(memory_len, batch_shape, mem_dim)
                ),
            ),
            ct.TensorType(name="num_obj_ptr_tokens", shape=(1,)),
        ],
        outputs=[ct.TensorType(name="fused_embedding")],
        minimum_deployment_target=min_target,
        compute_units=compute_units,
        compute_precision=precision,
    )

    validate_memory_attention(
        mlmodel,
        video_model,
        current_feat,
        current_pos,
        memory,
        memory_pos,
        int(num_obj_ptr_tokens.item()),
    )

    output_path = os.path.join(
        output_dir,
        f"SAM2{variant.fmt()}MemoryAttention{precision.value.upper()}",
    )
    mlmodel.save(output_path + ".mlpackage")


def export(
    output_dir: str,
    variant: SAM2Variant,
    points: Optional[List[Tuple[float, float]]],
    labels: Optional[List[int]],
    min_target: AvailableTarget,
    compute_units: ComputeUnit,
    precision: ComputePrecision,
):
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device("cpu")

    sam2_checkpoint = f"facebook/sam2.1-hiera-{variant.value}"

    with torch.no_grad():
        video_model = SAM2VideoPredictor.from_pretrained(
            sam2_checkpoint, device=device
        )
        video_model.eval()
        predictor = SAM2ImagePredictor(video_model)
        predictor.model.eval()

        orig_hw = export_image_encoder(
            predictor, variant, output_dir, min_target, compute_units, precision
        )
        export_points_prompt_encoder(
            predictor,
            variant,
            points,
            labels,
            orig_hw,
            output_dir,
            min_target,
            compute_units,
            precision,
        )
        export_mask_decoder(
            predictor, variant, output_dir, min_target, compute_units, precision
        )
        export_memory_encoder(
            video_model, variant, output_dir, min_target, compute_units, precision
        )
        export_memory_attention(
            video_model, variant, output_dir, min_target, compute_units, precision
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SAM2 Video -> CoreML CLI")
    parser = parse_args(parser)
    args = parser.parse_args()

    points, labels = None, None
    if args.points:
        points = [tuple(p) for p in ast.literal_eval(args.points)]
    if args.labels:
        labels = ast.literal_eval(args.labels)

    if points is None:
        raise ValueError("Points must be provided for tracing.")

    if labels is None:
        raise ValueError("Labels must be provided for tracing.")

    if not isinstance(points, list) or not all(
        isinstance(p, tuple) and len(p) == 2 for p in points
    ):
        raise ValueError("Points must be a tuple of 2D points")

    if not isinstance(labels, list) or not all(
        isinstance(l, int) and l in [0, 1] for l in labels
    ):
        raise ValueError("Labels must denote foreground (1) or background (0)")

    if len(points) != len(labels):
        raise ValueError("Number of points must match the number of labels")

    if len(points) > 16:
        raise ValueError("Number of points must be less than or equal to 16")

    export(
        args.output_dir,
        args.variant,
        points,
        labels,
        args.min_deployment_target,
        args.compute_units,
        args.precision,
    )
