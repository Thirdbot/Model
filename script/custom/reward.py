import ast
import re


class SegmentationReward:
    BBOX_PATTERN = re.compile(r"<bbox>\s*(\[.*?\])\s*</bbox>", re.DOTALL)
    REGION_PATTERN = re.compile(r"<region>.*?</region>", re.DOTALL)
    EVIDENCE_PATTERN = re.compile(r"<evidence>.*?</evidence>", re.DOTALL)
    ANSWER_PATTERN = re.compile(r"<answer>.*?</answer>", re.DOTALL)

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
            score += 0.25 if parsed["regions"] else 0.0
            score += 0.25 if parsed["bboxes"] else 0.0
            score += 0.2 if parsed["evidence_count"] > 0 else 0.0
            score += 0.15 if parsed["seg_count"] == len(parsed["regions"]) and parsed["seg_count"] > 0 else 0.0
            score += 0.15 if parsed["has_answer"] else 0.0
            rewards.append(score * self.format_weight)
        return rewards

    def bbox_iou_reward(self, completions, target=None, **kwargs):
        targets = self._resolve_targets(target, kwargs, len(completions))
        rewards = []
        for completion, expected in zip(completions, targets):
            pred_text = self._completion_to_text(completion)
            pred_bboxes = self._parse_prediction(pred_text)["bboxes"]
            target_bboxes = self._parse_prediction(str(expected))["bboxes"]

            if not pred_bboxes or not target_bboxes:
                rewards.append(0.0)
                continue

            best_iou = max(
                self._bbox_iou(pred_bbox, target_bbox)
                for pred_bbox in pred_bboxes
                for target_bbox in target_bboxes
            )
            rewards.append(best_iou * self.bbox_weight)
        return rewards

    def combined_reward(self, completions, target=None, **kwargs):
        format_scores = self.format_reward(completions, **kwargs)
        iou_scores = self.bbox_iou_reward(completions, target=target, **kwargs)
        return [format_score + iou_score for format_score, iou_score in zip(format_scores, iou_scores)]

    def _parse_prediction(self, text):
        bboxes = [
            bbox
            for bbox in self._extract_literals(text, self.BBOX_PATTERN)
            if self._valid_bbox(bbox)
        ]
        regions = self.REGION_PATTERN.findall(text)

        return {
            "bboxes": bboxes,
            "regions": regions,
            "evidence_count": len(self.EVIDENCE_PATTERN.findall(text)),
            "seg_count": text.count("<SEG>"),
            "has_answer": self.ANSWER_PATTERN.search(text) is not None,
        }

    @staticmethod
    def _extract_literals(text, pattern):
        values = []
        for match in pattern.finditer(text):
            try:
                values.append(ast.literal_eval(match.group(1)))
            except (SyntaxError, ValueError):
                continue
        return values

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
