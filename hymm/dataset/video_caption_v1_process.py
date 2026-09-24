
# =======================================================
# Caption process script used for new_caption_v1_zh and
# new_caption_v1_en
# new caption structure:
# {
#     "style_features": str,
#     "content_summary": str,
#     "background_audio": str,
#     "shots": [
#         {
#             "time_range": [start, end],
#             "static_description": str,
#             "dynamic_description": str,
#         }
#     ],
#     "tags": {...}
# }
# caption assembly:
# [style_features?] → content_summary → background_audio
#                       → shot_1 → shot_2 → ...
# each shot internally:
# [time_range?], static_description . dynamic_description
# time_range rules:
#   - len(shots) >= 2 → always include time_range for every shot
#   - len(shots) == 1 → include with probability caption_sample_ratio["time_range"]
# =======================================================

import json
import random
import re
from typing import Optional
from dataclasses import dataclass


@dataclass
class CaptionOut(object):
    caption: Optional[str] = ""
    lang: Optional[str] = ""
    key: Optional[str] = ""
    sel_col: Optional[str] = ""
    tag_keys: Optional[list[str]] = None

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        if not hasattr(self, key):
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
        setattr(self, key, value)

SEPARATORS = {
    "zh": {"comma": "，", "period": "。"},
    "en": {"comma": ", ", "period": ". "}
}


class CaptionAug:

    def __init__(self,
                 caption_sample_ratio=None,
                 ocr_only_long_caption=False,
                 num_replace_rate=0,
                 random_caption_tag_order=False,
                 logger=None,
                 ):
        """
        Args:
            caption_sample_ratio : 结构化caption采样比例
        """
        if logger is None:
            from loguru import logger
        self.logger = logger

        self.caption_sample_ratio = caption_sample_ratio
        if isinstance(caption_sample_ratio, str):
            self.caption_sample_ratio = json.loads(caption_sample_ratio)
        # content_summary / background_audio / static_description / dynamic_description
        # 是"必取"字段，对应概率只能是 1.0；缺省时自动填 1.0，显式配成非 1.0 直接报错。
        _MANDATORY_KEYS = (
            "background_audio",
            "static_description",
            "dynamic_description",
        )
        if isinstance(self.caption_sample_ratio, dict):
            for _k in _MANDATORY_KEYS:
                if _k in self.caption_sample_ratio and self.caption_sample_ratio[_k] != 1.0:
                    raise ValueError(
                        f"caption_sample_ratio['{_k}'] must be 1.0 (mandatory field), "
                        f"got {self.caption_sample_ratio[_k]!r}"
                    )
                self.caption_sample_ratio.setdefault(_k, 1.0)

        # Predefined keys
        self.caption_keys = {"style_features", "content_summary", "background_audio"}
        self.shot_keys = {"time_range", "static_description", "dynamic_description"}
        # tag_keys暂时不需要
        self.tag_keys = {"ip_tag", "audio_tag", "music_tag", "language_dialect_tag", "visual_realism", "overlay_tag"}
        self.predefined_keys = self.caption_keys.union(self.shot_keys, self.tag_keys)

        # User-defined caption sample ratio(csr) keys
        csr_keys = set([key for key, value in self.caption_sample_ratio.items() if value > 0])
        if csr_keys - self.predefined_keys:
            raise NotImplementedError(f"Unexpected keys in caption_sample_ratio: {csr_keys - self.predefined_keys}")
        self.valid_caption_keys = csr_keys.intersection(self.caption_keys)
        self.valid_shot_keys = csr_keys.intersection(self.shot_keys)
        self.valid_tag_keys = csr_keys.intersection(self.tag_keys)

        self.logger = logger
        self.logger.info(
            "CaptionAug using caption sample ratio: {}".format(json.dumps(self.caption_sample_ratio))
        )
        self.ocr_only_long_caption = ocr_only_long_caption
        self.num_replace_rate = num_replace_rate
        self.random_caption_tag_order = random_caption_tag_order

    @staticmethod
    def safe_load_string(text):
        """加载结构化caption数据, json or xml格式"""
        if isinstance(text, dict):
            return text
        try:
            text = json.loads(text)
        except Exception as e:
            raise NotImplementedError("json str format only {}, got str: {}".format(str(e), text))

        if type(text) is str:
            try:
                text = eval(text)
                if type(text) is str:
                    text = text.replace("'", "\"")
                    text = json.loads(text)
            except Exception as e:
                raise NotImplementedError("json str format only {}, got eval str: {}".format(str(e), text))

        if not isinstance(text, dict):
            print('load caption error', text)

        if "caption" in text:
            if isinstance(text["caption"], str):
                text = json.loads(text["caption"])
            else:
                text = text["caption"]

        # reload if still string
        if isinstance(text, str):
            try:
                text = json.loads(text)
            except Exception as e:
                raise NotImplementedError("json str format only {}, got str: {}".format(str(e), text))

        return text

    def random_replace_num(self, long_short_caption):
        """是否随机替换数字"""
        if self.num_replace_rate == 0 or self.ocr_in_caption(long_short_caption):
            return long_short_caption

        num_tag_map = {
            # "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",

            # "二": "2",
            # "三": "3",
            # "四": "4",
            # "五": "5",
            # "六": "6",
            # "七": "7",
            # "八": "8",
            # "九": "9",
            # "十": "10",
        }

        # 使用正则表达式匹配单词边界
        def replace_num(match):
            word = match.group(0).lower()
            if word in num_tag_map:
                # 对每个匹配到的数字按概率决定是否替换
                if random.random() < self.num_replace_rate:
                    return num_tag_map[word]
            return match.group(0)

        # 使用正则表达式替换，\b表示单词边界，这里中文匹配不生效
        pattern = r'\b(' + '|'.join(num_tag_map.keys()) + r')\b'
        new_caption = re.sub(pattern, replace_num, long_short_caption, flags=re.IGNORECASE)
        return new_caption

    @staticmethod
    def strip_zh(text):
        text = text.replace("。，", "，").replace("，。", "。").replace("。。", "。").replace("，，", "，")
        text = re.sub(r'\s+,', ' ', text).strip()
        text = text.lstrip("，。 ")
        return text

    @staticmethod
    def strip_en(text):
        text = text.replace(".,", ",").replace(",.", ".").replace("..", ".").replace(",,", ",")
        text = re.sub(r'\s+,', ' ', text).strip()
        text = text.lstrip(",. ")
        return text

    def ocr_in_caption(self, caption):

        pattern = r'\"(.*?)\"|“(.*?)”'
        matches = re.findall(pattern, caption)
        result = [match[0] or match[1] for match in matches]

        if result:
            return True
        return False

    # ------------------------------------------------------------------
    # time_range formatting
    # ------------------------------------------------------------------
    # Per-language candidate units for time values.
    _TIME_UNITS = {
        "zh": ("s", "秒"),
        "en": ("s",),
    }

    @staticmethod
    def _format_seconds(value, unit="s"):
        """Render a single time value with the given suffix.

        Preserves the original decimal precision of the input. 0 -> 0s / 0秒
        """
        if value is None:
            return ""
        s = str(value).strip()
        try:
            f = float(s)
        except (TypeError, ValueError):
            return f"{s}{unit}"
        if f == 0:
            return f"0{unit}"
        return f"{s}{unit}"

    def format_time_range(self, time_range, lang):
        """Sample a natural-language rendering of a [start, end] pair."""
        if not isinstance(time_range, (list, tuple)) or len(time_range) < 2:
            return ""

        unit = random.choice(self._TIME_UNITS.get(lang, ("s",)))
        start_s = self._format_seconds(time_range[0], unit=unit)
        end_s = self._format_seconds(time_range[1], unit=unit)
        if not start_s or not end_s:
            return ""

        if lang == "zh":
            styles = [
                f"[{start_s}~{end_s}]",
                f"[{start_s}-{end_s}]",
                f"({start_s}~{end_s})",
                f"[时间：{start_s}-{end_s}]",
                f"[时段：{start_s}-{end_s}]",
                f"[从{start_s}到{end_s}]",
                f"[{start_s}至{end_s}]",
            ]
        else:
            styles = [
                f"[{start_s}~{end_s}]",
                f"[{start_s}-{end_s}]",
                f"({start_s}~{end_s})",
                f"[time: {start_s}-{end_s}]",
                f"[period: {start_s}-{end_s}]",
                f"[from {start_s} to {end_s}]",
                f"[between {start_s} and {end_s}]",
            ]
        return random.choice(styles)

    # ------------------------------------------------------------------
    # Main compose
    # ------------------------------------------------------------------
    def random_compose(self, caption_dict, lang, return_key=False,
                       caption_keys=None, tag_keys=None, shot_keys=None,
                       ignore_tag_prob=False, caption_keys_prob=None):
        """

        Parameters
        ----------
        caption_dict: dict
        lang: str
            Language of the caption, either "zh" or "en".
        return_key: bool, optional
            Whether to return the key of the selected caption.
        caption_keys: list, optional
            Allow user to specify which caption keys to use.
        tag_keys: list, optional
            Allow user to specify which tag keys to use.
        shot_keys: list, optional
            Allow user to specify which shot keys to use.
        ignore_tag_prob: bool, optional
            Only valid when tag_keys is not None. If True, ignore the caption_sample_ratio for tag keys,
            and use all available tags specified by tag_keys.
        caption_keys_prob: dict, optional
            Allow user to specify the sampling probability of each caption key. (not tag keys)
        """
        assert lang in {"zh", "en"}, f"Unsupported language: {lang}"
        sep = SEPARATORS[lang]

        if caption_keys is not None:
            assert isinstance(caption_keys, list), \
                f"caption_keys must be a list: {caption_keys}, got {type(caption_keys)}"
            valid_caption_keys = caption_keys
        else:
            valid_caption_keys = self.valid_caption_keys
        if tag_keys is not None:
            assert isinstance(tag_keys, list), \
                f"tag_keys must be a list: {tag_keys}, got {type(tag_keys)}"
            valid_tag_keys = tag_keys
        else:
            valid_tag_keys = self.valid_tag_keys
        if shot_keys is not None:
            assert isinstance(shot_keys, list), \
                f"shot_keys must be a list: {shot_keys}, got {type(shot_keys)}"
            valid_shot_keys = shot_keys
        else:
            valid_shot_keys = self.valid_shot_keys

        def _prob(key):
            if caption_keys_prob is not None and key in caption_keys_prob:
                return caption_keys_prob[key]
            return self.caption_sample_ratio.get(key, 0.0)

        # tags暂时不需要，先组装类似global caption，然后组装shots_text
        global_caption_values = {}
        tag_candidates = {}
        for key in caption_dict:
            if key in valid_caption_keys:
                if caption_dict[key] and caption_dict[key] != "" and caption_dict[key].lower() != "none" and caption_dict[key] != "无":
                    if random.random() < _prob(key):
                        global_caption_values[key] = caption_dict[key].strip()
            elif key in valid_tag_keys:
                if random.random() < self.caption_sample_ratio[key]:
                    tag = caption_dict[key].strip()
                    if tag != "" and tag.lower() != "none" and tag != "无":
                        tag_candidates[key] = tag.strip()

        # 单镜头下，time_range是可选的（可以指定概率）
        # 多镜头，time_range是必须的，即使指定选择time_range的概率不是1，也要包含time_range
        shots = caption_dict.get("shots") or []
        if len(shots) == 0:
            raise ValueError("shots is required but empty.")

        num_time_ranges = sum(
            1 for s in shots if isinstance(s, dict) and s.get("time_range")
        )
        is_multi_shot = num_time_ranges > 1
        include_time_range = (
            is_multi_shot
            or (
                "time_range" in valid_shot_keys
                and random.random() < _prob("time_range")
            )
        )

        shot_parts = []
        for shot in shots:
            if not isinstance(shot, dict):
                continue

            descs = []
            static_desc = (shot.get("static_description") or "").strip()
            if static_desc:
                descs.append(static_desc)
            dynamic_desc = (shot.get("dynamic_description") or "").strip()
            if dynamic_desc:
                descs.append(dynamic_desc)

            if not descs:
                continue  # skip degenerate shot with no description

            timestamp_str = ""
            if include_time_range:
                timestamp_str = self.format_time_range(shot.get("time_range"), lang)

            if timestamp_str:
                shot_text = timestamp_str + sep["comma"] + sep["period"].join(descs)
            else:
                shot_text = sep["period"].join(descs)

            shot_parts.append(shot_text)

        if not shot_parts:
            raise ValueError("No valid shot descriptions in shots.")

        shots_text = sep["period"].join(shot_parts)

        caption_order = ("style_features", "content_summary", "background_audio")
        ordered = [global_caption_values[k] for k in caption_order if k in global_caption_values]
        ordered.append(shots_text)

        caption = sep["period"].join(ordered)
        caption = self.strip_zh(caption) if lang == "zh" else self.strip_en(caption)
        caption = caption.strip().replace("\\N", "").strip("，,")

        if return_key:
            # key之前是selected_key，现在改成new_caption_v1
            return CaptionOut(
                caption=caption,
                lang=lang,
                key="new_caption_v1",
                tag_keys=[],
            )
        return caption

    def caption_aug(self, raw_string: str | dict, lang, **kwargs):
        caption_dict = self.safe_load_string(raw_string)

        if kwargs.get("return_key"):
            out = self.random_compose(caption_dict, lang, return_key=True)
            out.caption = self.random_replace_num(out.caption)
            return out

        caption = self.random_compose(caption_dict, lang)
        return self.random_replace_num(caption)


if __name__ == "__main__":

    test_input = {
        "style_features": "Interview, realistic, somber and heavy atmosphere",
        "content_summary": "An elderly white woman is being interviewed indoors, her expression sorrowful as she recalls a past event in Dutch, describing people leaving and a supervisor intervening.",
        "shots":[
            {
                "time_range":["0", "9.0"],
                "static_description": "Close-up, eye-level view, softly lit with frontal lighting. The frame features an elderly white woman with short gray curly hair, blue eyes, deeply lined face, flushed cheeks, and red lipstick. She wears a black coat and a red scarf with a white floral pattern, a black microphone clipped to her collar. She sits in front of a brown wooden piece of furniture, with a pale yellow textured wall behind her, on which hangs a miniature model depicting a canal and Dutch-style buildings on both sides.",
                "dynamic_description": "Initially, the elderly white woman faces the camera, lips moving, speaking in a hoarse voice with a heavy expression<speech>weg haden ze nog gezegd, ze liepen weg.</speech>. Then, she briefly lowers her head, her gaze moving downward, mouth slightly open as if sighing. Immediately afterward, she raises her head to look at the camera again, continuing<speech>Nou, dan kwam de directie er bij, want de directeur moest oepen.</speech>, blinking slowly throughout, her expression solemn."
            }
        ],
        "background_audio": "No background music, quiet recording environment, close-mic capture, no noticeable reverb.",
        "tags":{
            "ip_tag": "未知",
            "audio_tag": "多声道",
            "music_tag": "无",
            "language_dialect_tag": "荷兰语",
            "visual_realism": "写实",
            "overlay_tag": "无"
        }
    }

    caption_sample_ratio = {
        "style_features": 0.0,
        "content_summary": 0.0,
        "background_audio": 1.0,
        "static_description": 1.0,
        "dynamic_description": 1.0,
        "time_range": 0.0,
    }

    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("CaptionAug")

    aug = CaptionAug(caption_sample_ratio=caption_sample_ratio, logger=logger)

    print("=" * 80)
    print("  Input caption dict:")
    print("=" * 80)
    print(json.dumps(test_input, ensure_ascii=False, indent=2))

    print("\n" + "=" * 80)
    print("  Output (return_key=True):")
    print("=" * 80)
    out = aug.caption_aug(test_input, lang="en", return_key=True)
    print(f"  key:     {out.key}")
    print(f"  lang:    {out.lang}")
    print(f"  caption:\n{out.caption}")

    print("\n" + "=" * 80)
    print("  Output (return_key=False, plain string):")
    print("=" * 80)
    caption = aug.caption_aug(test_input, lang="en")
    print(caption)
