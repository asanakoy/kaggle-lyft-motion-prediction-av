INPUT_DIR="./input"
if [ ! -d "$INPUT_DIR/scenes/test.zarr" ]; then
    pushd "$INPUT_DIR"
    echo "Downloading dataset (train + val + test) from Kaggle..."
    kaggle competitions download -c lyft-motion-prediction-autonomous-vehicles
    echo "Unpacking..."
    unzip lyft-motion-prediction-autonomous-vehicles.zip
    popd
fi

cd src/1st_level
python prerender_raster.py --action render --dset_name test --scene_step 16 --initial_scenes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 --dir_name pre_render_h01248_XXL
