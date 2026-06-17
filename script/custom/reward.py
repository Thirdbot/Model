import ast
import re


class SegmentationReward:
    BBOX_PATTERN = re.compile(r"<bbox>\s*(\[.*?\])\s*</bbox>", re.DOTALL)
    POINTS_PATTERN = re.compile(r"<points>\s*(\[.*?\])\s*</points>", re.DOTALL)
    LABELS_PATTERN = re.compile(r"<labels>\s*(\[.*?\])\s*</labels>", re.DOTALL)

    def __init__(self, format_weight=1.0, bbox_weight=2.0, empty_penalty=-1.0):
        self.format_weight = format_weight
        self.bbox_weight = bbox_weight
        self.empty_penalty = empty_penalty

    def format_reward(self, completions, **kwargs):
        rewards = []
        for completion in completions:
            text = self._completion_to_text(completion)
            if not text.strip():
                rewards.append(self.empty_penalty)
                continue

            score = 0.0
            parsed = self._parse_prediction(text)
            score += 0.4 if parsed["bbox"] is not None else 0.0
            score += 0.25 if parsed["points"] is not None else 0.0
            score += 0.25 if parsed["labels"] is not None else 0.0
            score += 0.1 if parsed["valid"] else 0.0
            rewards.append(score * self.format_weight)
        return rewards

    def bbox_iou_reward(self, completions, target=None, **kwargs):
        targets = self._resolve_targets(target, kwargs, len(completions))
        rewards = []
        for completion, expected in zip(completions, targets):
            pred_text = self._completion_to_text(completion)
            pred_bbox = self._parse_prediction(pred_text)["bbox"]
            target_bbox = self._parse_prediction(str(expected))["bbox"]

            if pred_bbox is None or target_bbox is None:
                rewards.append(0.0)
                continue
            rewards.append(self._bbox_iou(pred_bbox, target_bbox) * self.bbox_weight)
        return rewards

    def combined_reward(self, completions, target=None, **kwargs):
        format_scores = self.format_reward(completions, **kwargs)
        iou_scores = self.bbox_iou_reward(completions, target=target, **kwargs)
        return [format_score + iou_score for format_score, iou_score in zip(format_scores, iou_scores)]

    def _parse_prediction(self, text):
        bbox = self._extract_literal(text, self.BBOX_PATTERN)
        points = self._extract_literal(text, self.POINTS_PATTERN)
        labels = self._extract_literal(text, self.LABELS_PATTERN)

        bbox = bbox if self._valid_bbox(bbox) else None
        points = points if isinstance(points, list) else None
        labels = labels if isinstance(labels, list) else None

        return {
            "bbox": bbox,
            "points": points,
            "labels": labels,
            "valid": bbox is not None and points is not None and labels is not None,
        }

    @staticmethod
    def _extract_literal(text, pattern):
        match = pattern.search(text)
        if match is None:
            return None
        try:
            return ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            return None

    @staticmethod
    def _valid_bbox(value):
        if not isinstance(value, list) or len(value) != 4:
            return False
        return all(isinstance(number, (int, float)) for number in value)

    @staticmethod
    def _bbox_iou(pred, target):
        px1, py1, px2, py2 = pred
        tx1, ty1, tx2, ty2 = target

        inter_x1 = max(px1, tx1)
        inter_y1 = max(py1, ty1)
        inter_x2 = min(px2, tx2)
        inter_y2 = min(py2, ty2)

        inter_width = max(0.0, inter_x2 - inter_x1)
        inter_height = max(0.0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height

        pred_area = max(0.0, px2 - px1) * max(0.0, py2 - py1)
        target_area = max(0.0, tx2 - tx1) * max(0.0, ty2 - ty1)
        union = pred_area + target_area - intersection

        if union <= 0:
            return 0.0
        return intersection / union

    @staticmethod
    def _completion_to_text(completion):
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list) and completion:
            first = completion[0]
            if isinstance(first, dict):
                return str(first.get("content", ""))
        if isinstance(completion, dict):
            return str(completion.get("content", ""))
        return str(completion)

    @staticmethod
    def _resolve_targets(target, kwargs, expected_length):
        targets = target or kwargs.get("targets") or kwargs.get("solution") or kwargs.get("answer")
        if targets is None:
            return [""] * expected_length
        if isinstance(targets, str):
            return [targets] * expected_length
        return targets


segmentation_reward = SegmentationReward()
format_reward = segmentation_reward.format_reward
bbox_iou_reward = segmentation_reward.bbox_iou_reward
combined_reward = segmentation_reward.combined_reward
