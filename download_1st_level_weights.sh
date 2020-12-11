dst_dir=src/1st_level/checkpoints
echo "Downloading all 1st level pretrained weights in ${dst_dir}"
if [ ! -d "$dst_dir" ]; then
    mkdir -p $dst_dir
fi
cd "$dst_dir"

#wget -O weights.zip https://www.dropbox.com/sh/6809fs9o2b3ppjp/AAACJBBoSPqFtB5QwL0iX6KBa/output/checkpoints?dl=0&subfolder_nav_tracking=1
echo "Unpacking..."
unzip weights.zip
echo "===========================+"
echo "${dst_dir}:"
ls -al
