# ObPose
## Setup
For video data:

Please put the downloaded checkpoints to "./checkpoints/MoveObjs_obsurf/" folder

Please put the downloaded data file "raw" to "./data/MoveObjs_obsurf/" folder

For static image data:

Please put the downloaded checkpoints to "./checkpoints/obj_tabel_multiview_obsurf/" folder

Please put the downloaded data file "raw" to "./data/obj_tabel_multiview/" folder

## Training from scratch.
```bash
python train_obsurf.py --ex_id 0 --bs 32 --seed 20 --ray_sample
python train_obsurf.py --ex_id 1 --bs 32 --seed 37 --ray_sample
python train_obsurf.py --ex_id 2 --bs 32 --seed 682 --ray_sample
```

```bash
python train_obsurf_video.py --ex_id 0 --bs 32 --seed 20 --ray_sample
python train_obsurf_video.py --ex_id 1 --bs 32 --seed 37 --ray_sample
python train_obsurf_video.py --ex_id 2 --bs 32 --seed 682 --ray_sample
```

## Evaluation
```bash
python eval_seg_obsurf.py --ex_id 0 --it 90000
python eval_seg_obsurf.py --ex_id 1 --it 90000
python eval_seg_obsurf.py --ex_id 2 --it 90000
```

```bash
python eval_seg_obsurf_static.py --ex_id 0 --it 100500
python eval_seg_obsurf_static.py --ex_id 1 --it 100500
python eval_seg_obsurf_static.py --ex_id 2 --it 100500
```

