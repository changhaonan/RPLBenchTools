# Dataset management

1. Objaverse Dataset.
We need to download some data from objevarse dataset. 
```
python process_objaverse.py
```

2. Prepare PartNet-Mobility data.

Download the partnet-mobility data at: https://sapien.ucsd.edu/. Extract the data to `raw_data`.

2.1. If you want to use the exported objects in Sapien, use this:
`python lgmcts/urdf_fixer.py --data_dir=./raw_data/dataset`.

2.2. If you want to use the generated objs in Ominisim or IsaacLab, using this:
`python lgmcts/urdf_deep_fixer.py --data_dir=./raw_data/dataset`.

run the `python lgmcts/urdf_deep_fixer.py --data_dir=./raw_data/dataset`.

Afer download, run the `process_partnet.py`. Set `partnet_data_dir` to be your downoaded part.


3. Check the generated objs.

```
python vis_obj.py --id=12530 --joint_values="0,1" --assets_dir="./assets_v2"
```

If will show `Structure` and `Pyrender` sequentially. Press `Q` to switch.

## Trouble shot

1. Some objects, such as `102001`, `102018` has a handle, but the handle's name is not handle.
2. Create `12349`.
3. Objects such as `27267` has a bug.
4. `34178` has a bug, for object that is not initialized at `0`. All has a bug.
5. `12071` has object that has shape 0.