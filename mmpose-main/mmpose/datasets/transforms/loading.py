# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional

import numpy as np
from mmcv.transforms import LoadImageFromFile

from mmpose.registry import TRANSFORMS
@TRANSFORMS.register_module()
class LoadImage(LoadImageFromFile):
    def transform(self, results: dict) -> Optional[dict]:
        try:
            if 'img' not in results:
                results = super().transform(results)
            else:
                img = results['img']
                assert isinstance(img, np.ndarray)
                if self.to_float32:
                    img = img.astype(np.float32)

                if 'img_path' not in results:
                    results['img_path'] = None
                results['img_shape'] = img.shape[:2]
                results['ori_shape'] = img.shape[:2]
        except Exception as e:
            print(f'[警告] 跳过缺失图像: {results.get("img_path", "unknown")} - {e}')
            return None  # ⭐️ 返回 None 跳过这一项

        return results