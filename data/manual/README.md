# Manual / ITB Stereo Dataset Splits

The EFSNet manual-data configs expect split files in this directory:

- `zed_file_train.txt`
- `zed_file_val.txt`
- optionally `zed_file_all.txt`

Each line should contain three paths relative to `DATA_PATH`:

```text
left_image_path right_image_path disparity_numpy_path
```

The disparity file is loaded with `numpy.load`, so it should be a `.npy` disparity map.
