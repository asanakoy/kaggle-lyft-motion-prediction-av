INPUT_DIR="./input/scenes"
if [ ! -d "$INPUT_DIR" ]; then
    mkdir -p $INPUT_DIR
fi
pushd $INPUT_DIR
echo "Downloading Full Training dataset (train_XXL) from Kaggle..."
kaggle datasets download philculliton/lyft-full-training-set
echo "Unpacking..."
unzip lyft-full-training-set.zip
mv train_full.zarr train_XXL.zarr
popd

cd src/1st_level
python prerender_raster.py --dset_name train_XXL --scene_step 32 --skip_frame_step 7 --initial_scenes 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 --num_jobs 32 --dir_name pre_render_h01248_XXL
