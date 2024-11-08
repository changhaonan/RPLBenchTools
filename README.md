# Dataset management

<!-- 1. Objaverse Dataset.
We need to download some data from objevarse dataset. 
```
python process_objaverse.py 
```-->

1. Prepare PartNet-Mobility data. Download the partnet-mobility data at: https://sapien.ucsd.edu/. Extract the data to `raw_data`. Set `partnet_data_dir` to be your downoaded part.

1.1. If you want to use the exported objects in Sapien, use this:
```
python lgmcts/urdf_fixer.py --data_dir=./raw_data/dataset
```

1.2. If you want to use the generated objs in Ominisim or IsaacLab, using this:
```
python lgmcts/urdf_deep_fixer.py --data_dir=./raw_data/dataset
```

This will remove some textures.

2. Run processing code.
```
python process_partnet.py
```

3. Check the generated objs.

```
python vis_obj.py --id=12530 --joint_values="0,1" --assets_dir="./assets_v2"
```

If will show `Structure` and `Pyrender` sequentially. Press `Q` to switch.

## Trouble-shot

1. Some objects, such as `102001`, `102018` has a handle, but the handle's name is not handle.
2. Create `12349`.
3. Objects such as `27267` has a bug.
4. `34178`, `7265` has a bug, for object that is not initialized at `0`. All has a bug.
5. `12071` has object that has shape 0.
6. `22301` seems to be have a little problem.