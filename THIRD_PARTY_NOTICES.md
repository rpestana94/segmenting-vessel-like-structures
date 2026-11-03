# Third-party notices

This project reuses and adapts code from the following open-source projects.
Their licences are permissive and compatible with reuse provided attribution
and licence notices are retained.

| Component | Upstream project | Licence | Use in this repo |
|---|---|---|---|
| U-Net segmentation | [jocicmarko/ultrasound-nerve-segmentation](https://github.com/jocicmarko/ultrasound-nerve-segmentation) | MIT | architecture/training pattern (header retained in historical scripts) |
| Medical image segmentation workflow | [CVxTz/medical_image_segmentation](https://github.com/CVxTz/medical_image_segmentation) (Mansar) | MIT | segmentation pipeline reference |
| ESPCN super-resolution | [keras-team/keras-io](https://github.com/keras-team/keras-io) — `examples/vision/super_resolution_sub_pixel.py` (X. Long) | Apache-2.0 | ESPCN model + sub-pixel upscaling |

The ESPCN architecture itself is from Shi et al. (2016), "Real-Time Single Image
and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural
Network" (arXiv:1609.05158). The U-Net architecture is from Ronneberger et al.
(2015), arXiv:1505.04597.

Copies of the upstream MIT and Apache-2.0 licence texts should accompany this
file in a `licenses/` directory before public release.
