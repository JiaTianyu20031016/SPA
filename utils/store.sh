src=/root/jiaty/projects/SPA/alignment-handbook
dst=/dev/jiaty/zephyr-spa-zero-suffix-1024-ArmoRM-epoch2

if [ -d "${dst}" ]; then
  echo "${dst} already exist."
  exit
else
  mkdir ${dst}
fi

cp -r ${src}/save_model ${dst}/save_model
cp -r ${src}/datasets ${dst}/datasets
cp ${src}/output.log ${dst}/output.log
cp -r ${src}/save_confidence ${dst}/save_confidence
