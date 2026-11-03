# DRIVE — retinal vessel segmentation (training images 21–40)

Digital Retinal Images for Vessel Extraction. Each image has a 1st-manual vessel annotation (ground truth) and a FOV mask. Used by Item 5 (Dice/IoU/AUC) and the timing benchmark (Item 8).

## Download
- **Source:** https://drive.grand-challenge.org/  (registration required)

> **Licence / redistribution:** The DRIVE dataset does not carry a clearly specified open licence (reported variously as unspecified / unknown / non-commercial research use), and its official distribution requires accepting the site's terms before download. Because no licence explicitly grants redistribution, the images are **NOT bundled in this repository** — download them from the official site and place the files as listed below. Filenames here match the canonical DRIVE distribution.

**Used by:** reproducir_todo.sh (Item 5), reproduce_paper.py, run_demo.sh, entrenar_base_reproducible.sh, reproduce_timing.py (Item 8)

## Files (60 files, 14.4 MB) — place them in `data/DRIVE_training/`

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `1st_manual/21_manual1.gif` | 10243 | `8b91ecd3ad91b3d7111f8f22e29913eb69502f8bfd4a1339db6508d73f52b8bb` |
| `1st_manual/22_manual1.gif` | 12416 | `3e24a5eab43676e8f9a7149984e4c7477f3b3829c4a5241d733eb43719e67967` |
| `1st_manual/23_manual1.gif` | 8086 | `f9fc8da936de5e21effbd66b76f33ea6c529661ad8c49b4314c6a89d907461a0` |
| `1st_manual/24_manual1.gif` | 13349 | `ed400fb8471f272ad5e9d2693336cbedb7c437e299b918edeae8498d07c647fb` |
| `1st_manual/25_manual1.gif` | 11472 | `64410317c2f901643da575c7871dc972c9585220466cd81545d30ced922d5159` |
| `1st_manual/26_manual1.gif` | 10361 | `e42498f4273217982cd90536d9900cc757a62c8358af0485df716fa89d04407c` |
| `1st_manual/27_manual1.gif` | 12014 | `a38ad76f19ef68bce461b493f9987142e9185982ee5517ff97cdbb57b7a57420` |
| `1st_manual/28_manual1.gif` | 12427 | `7469ed6208d32bc150bf66866fdc11d548201585ca32eec3b5130e9cffebb8a7` |
| `1st_manual/29_manual1.gif` | 11520 | `6e91d5b87ab1a73f6f510e86745307a5e0e5af0ca9b8e629e947cc30f4caca07` |
| `1st_manual/30_manual1.gif` | 10564 | `bdad759b5f833156dcf52115ba77736706d50be3e1d6e44adaf6ef0170d1a4a8` |
| `1st_manual/31_manual1.gif` | 8834 | `b90c0fd6986a7f76861c4d4b2d07aef013f98d4b1ed28d844acbbf81c99de2c4` |
| `1st_manual/32_manual1.gif` | 11002 | `004e501032efe825785b4711ba32fe150a17cede04d37f416961006a0cf84e3f` |
| `1st_manual/33_manual1.gif` | 11285 | `5d49c2449fcd341cb762928cad0dbcbca4e695a48643152f1acc8c5670d3add9` |
| `1st_manual/34_manual1.gif` | 11319 | `164ecee52bd14ddfe5233ab88b68a1f07399f82428588bfb3866de0f93c0a9f2` |
| `1st_manual/35_manual1.gif` | 11076 | `cada33fec4888c95e1bcbde3df7755ef1e289b63c047a57e0ff2d37d04be7168` |
| `1st_manual/36_manual1.gif` | 13145 | `a048905f16a1c4ed3c62dbc32fad928b13694d17b325352fff03541f852e0045` |
| `1st_manual/37_manual1.gif` | 11163 | `95012bee9015a712d3d4014220e4f21eef78dcd942a8acc2fa321b5f8c3bbc4a` |
| `1st_manual/38_manual1.gif` | 11493 | `a216ebf019ad018dc897b9c22831f9c428eb51fba73379aa9463001db09846ca` |
| `1st_manual/39_manual1.gif` | 11650 | `3440e0be4dce8ec83494d4e0b07a57db9575c0312023d4c98d395f5d217ff7a8` |
| `1st_manual/40_manual1.gif` | 10663 | `3f352ec665f77caab4fe806019169d222c1403281ccb9a9be9bf2070fa5c8e89` |
| `images/21_training.tif` | 758672 | `e7af55375ba088ac63db120377eaf94b2405fe80b7874d6e1d236cda4f8c4c83` |
| `images/22_training.tif` | 751306 | `4fc9bd6642ab205174d4609c10836b2831f3f8aefa5febac1f136273ae4eebd6` |
| `images/23_training.tif` | 654844 | `4d20167c45d4e495bf1aac435c7f611a991fafdbcba8f39f5b2e667f3809c64b` |
| `images/24_training.tif` | 695020 | `081db8928e07c9a222b05596f54f60770f37c1cdb39308377b420dc58a49c361` |
| `images/25_training.tif` | 722436 | `544904428f4286cfde44f2f24171c59f1fbc86c00833a4862f31f7eba3c71841` |
| `images/26_training.tif` | 701824 | `97b0b7962cbdb6c05c8dc090ef1fe11f4bcc7dda024471ea7521871d97454dfe` |
| `images/27_training.tif` | 730206 | `4ebfe2e985b9459b2ea9fd46d61e608c53a0044217f99126ea0a5706accce9e1` |
| `images/28_training.tif` | 754614 | `3ce704a1bc9d1cd82d27817db9ef5e30f030986bb209612139b998792860fbeb` |
| `images/29_training.tif` | 739372 | `b0dc412fbbe208c47134208f617ec0b630cb6adba794e384fa1b7f9b8217cbc3` |
| `images/30_training.tif` | 703530 | `713d1764d784ac3351a0ceb18051163991d0b222be875430e7ba54b05b333ac0` |
| `images/31_training.tif` | 773854 | `89a2606a6b1f226c53304e7c4148463378e7c8c94631418058812f072a00e7c6` |
| `images/32_training.tif` | 739210 | `915f74c3b2512ac842339a82be1e5d26d05ce3f9ee591c849606c9fe3dde4dca` |
| `images/33_training.tif` | 757826 | `93d377c99a50dd6c0b08c6ff55b4d32bc43cba2b7d65a286bfd0747c4a189a87` |
| `images/34_training.tif` | 818982 | `9c7bd663458a7a8e289c414f624ebc9c8eff672a703a394968fd0e7853eaaafa` |
| `images/35_training.tif` | 798744 | `de7affead453633b961a8e2525a3a5f22c4d432413ece9480fa3301dffeb9675` |
| `images/36_training.tif` | 747538 | `359626ded5d58eb8c6aa0d71f280e4c90a66189d9d10d8d871d5f3caf7f21929` |
| `images/37_training.tif` | 784176 | `4e741535cce29ed5c28ff1f3e51316c3efc471dcbc1ab794a71c0852bdc04e8f` |
| `images/38_training.tif` | 730086 | `bfbfe78a1015abd460f6d423a3d69d0cd41c38983502f5c6557202e4ff5e06a3` |
| `images/39_training.tif` | 749418 | `5b6a396282615674d0dee65c504ec1ea860a71e127802c743ef3bda9705f9296` |
| `images/40_training.tif` | 692866 | `cfaf945b07dc28dc3efe972e0ac353a04053db86f4e8e02ba95b0e1859e8acdb` |
| `mask/21_training_mask.gif` | 3306 | `acf23eeef91bd7c59624b9c2bb014703bcb37ef65cc8da2c0d221f3ca43dadc7` |
| `mask/22_training_mask.gif` | 3309 | `067f41ac14c4e08fa981e24baa16f11f1044f8479ba750e7ff282a66e6aebe28` |
| `mask/23_training_mask.gif` | 3319 | `68fb437ac43d39aba36e180f5ce6bee17950b355475744650fc88e8b23ef4a40` |
| `mask/24_training_mask.gif` | 3312 | `e2d16844e3f895cb9dffe8c568c34fb75c1fc02ef01bd072c8dd3199f85e655c` |
| `mask/25_training_mask.gif` | 3301 | `caf18b23eb9f52fc0f214daa74c876b817aed1184192b9a0a99c75f3475acb24` |
| `mask/26_training_mask.gif` | 3300 | `ea9358a2ff36b865b7971ccdc55e39098e3540845e9981e20c445b39c160a759` |
| `mask/27_training_mask.gif` | 3306 | `bce99f57f8125a4855dcbcd857f42bea59a076cc03368e2c7a41ab39d577fba7` |
| `mask/28_training_mask.gif` | 3307 | `0da824999724bcf89dd522f0657082e4aa6be1dee86c72c3feee4c1d0ea01aff` |
| `mask/29_training_mask.gif` | 3300 | `2c2140dcb15bb4a3a0a855ec1f12c3be3fc64439ffb3db5c083ba42221b6b66a` |
| `mask/30_training_mask.gif` | 3306 | `d5f71ebfff75c5ccd8657400b144d047379c3aa4223c0a359540a699d32848e5` |
| `mask/31_training_mask.gif` | 3304 | `a09e85cc750a199827a5c12e960b2887336d344504188dbe97ae4334331caeea` |
| `mask/32_training_mask.gif` | 3303 | `588b55ba3cb831ec502e54e25762a716ef384060abc5a947309633045849e7cf` |
| `mask/33_training_mask.gif` | 3312 | `5cafca1812bf58ddc4799f88a89a8535112a3158d59a4f6726cc90546965ca91` |
| `mask/34_training_mask.gif` | 3303 | `6deb28fbe64d29a44b5550726d59e9a9abb6225029ab6e80e582ac0c5eaf1ca4` |
| `mask/35_training_mask.gif` | 3309 | `eb0634285e3debe62fee8e981e7771cdb5ef1b5a961697ab11182f91e6795795` |
| `mask/36_training_mask.gif` | 3312 | `8a3b9fc71d30f5fec633ea310740df6a8c9b0f08d9da5561060b7be9601afae9` |
| `mask/37_training_mask.gif` | 3307 | `8cae242587defaa4866f740bd8fd980b34064afdec10c1e36a37cc7b1fa196c8` |
| `mask/38_training_mask.gif` | 3304 | `a915626cf9832e013ba5c4d8cb3f54613638846f630e6f49b239bdd22e93ab47` |
| `mask/39_training_mask.gif` | 3307 | `de5d7af3e3a5c7ce0c0d387253728ee867c4ff9b6c23b893d9adb7a046b86ff1` |
| `mask/40_training_mask.gif` | 3306 | `9c70f7d760dd8afee324a80a163bda9b9982e91cdad4a4818c42d5b06e0729bc` |

After downloading, verify integrity against the SHA-256 column above (e.g. `sha256sum data/DRIVE_training/<file>`). These are the exact files the scripts expect.