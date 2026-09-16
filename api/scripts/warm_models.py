"""Download the OCR weights at image build time so containers never fetch at runtime.

Run as a Docker build step. Keep the model set in sync with ocr.py's defaults.
"""

import numpy as np
from paddleocr import PaddleOCR

ocr = PaddleOCR(
    lang="en",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=True,
)
# One tiny predict forces lazy sub-model init (det, rec, textline orientation) so every
# weight file is resolved and cached now rather than on the first real page.
ocr.predict(np.full((64, 256, 3), 255, dtype=np.uint8))
print("OCR weights cached")
