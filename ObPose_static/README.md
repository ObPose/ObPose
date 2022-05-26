# ObPose
## Setup
Please put the downloaded data file "raw" to "./data/obj_tabel_multiview/" folder

Please put the downloaded checkpoints "raw" to "./checkpoints/obj_tabel_multiview/" folder

Install the point operations for KPConv
```bash
cd lib/pointops
python3 setup.py install
```

## Training from scratch.
```bash
python train.py --ex_id 0 --bs 8 --seed 20
python train.py --ex_id 1 --bs 8 --seed 63
python train.py --ex_id 2 --bs 8 --seed 422
```

```bash
python train_with_rot.py --ex_id 3 --bs 8 --seed 20
python train_with_rot.py --ex_id 4 --bs 8 --seed 63
python train_with_rot.py --ex_id 5 --bs 8 --seed 422
```

```bash
python train_icsbp.py --ex_id 6 --bs 8 --seed 20
python train_icsbp.py --ex_id 7 --bs 8 --seed 63
python train_icsbp.py --ex_id 8 --bs 8 --seed 422
```

```bash
python train_slot.py --ex_id 9 --bs 8 --seed 20
python train_slot.py --ex_id 10 --bs 8 --seed 63
python train_slot.py --ex_id 11 --bs 8 --seed 422
```
## Evaluation
```bash
python eval_seg.py --ex_id 0 --it 100500
python eval_seg.py --ex_id 1 --it 100500
python eval_seg.py --ex_id 2 --it 100500
```

```bash
python eval_seg_with_rot.py --ex_id 3 --it 100500
python eval_seg_with_rot.py --ex_id 4 --it 100500
python eval_seg_with_rot.py --ex_id 5 --it 100500
```

```bash
python eval_seg_icsbp.py --ex_id 6 --it 100500
python eval_seg_icsbp.py --ex_id 7 --it 100500
python eval_seg_icsbp.py --ex_id 8 --it 100500
```

```bash
python eval_seg_slot.py --ex_id 9 --it 100500
python eval_seg_slot.py --ex_id 10 --it 100500
python eval_seg_slot.py --ex_id 11 --it 100500
```



