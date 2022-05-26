# ObPose
## Setup
Please put the downloaded checkpoints to "./checkpoints/MoveObjs/" folder

Please put the downloaded data file "raw" to "./data/MoveObjs/" folder

Install the point operations for KPConv
```bash
cd lib/pointops
python3 setup.py install
```

## Training from scratch.
```bash
python train.py --ex_id 0 --detach --bs 2 --seed 20
python train.py --ex_id 1 --detach --bs 2 --seed 37
python train.py --ex_id 2 --detach --bs 2 --seed 682
```

```bash
python train_slot.py --ex_id 3 --bs 2 --seed 20
python train_slot.py --ex_id 4 --bs 2 --seed 37
python train_slot.py --ex_id 5 --bs 2 --seed 682
```

```bash
python train_with_rot.py --ex_id 6 --detach --bs 2 --seed 20
python train_with_rot.py --ex_id 7 --detach --bs 2 --seed 37
python train_with_rot.py --ex_id 8 --detach --bs 2 --seed 682
```

```bash
python train_icsbp.py --ex_id 9 --detach --bs 8 --seed 20
python train_icsbp.py --ex_id 10 --detach --bs 8 --seed 37
python train_icsbp.py --ex_id 11 --detach --bs 8 --seed 682
```

## Evaluation
```bash
python eval_seg.py --ex_id 0 --it 100000
python eval_seg.py --ex_id 1 --it 100000
python eval_seg.py --ex_id 2 --it 100000
```

```bash
python eval_seg_slot.py --ex_id 3 --it 100000
python eval_seg_slot.py --ex_id 4 --it 100000
python eval_seg_slot.py --ex_id 5 --it 100000
```

```bash
python eval_seg_with_rot.py --ex_id 6 --it 100000
python eval_seg_with_rot.py --ex_id 7 --it 100000
python eval_seg_with_rot.py --ex_id 8 --it 100000
```

```bash
python eval_seg_icsbp.py --ex_id 9 --it 100000
python eval_seg_icsbp.py --ex_id 10 --it 100000
python eval_seg_icsbp.py --ex_id 11 --it 100000
```
