dst=/root/jiaty/projects/SPA/alignment-handbook
src=/dev/jiaty/zephyr-spa-zero-suffix-1024-avg

if [ -d "${dst}" ]; then
  if [ -d "${dst}/save_model" ]; then
    rm -rf ${dst}/save_model
  fi
  cp -r ${src}/save_model ${dst}/save_model
  
  if [ -d "${dst}/datasets" ]; then
    rm -rf ${dst}/datasets
  fi
  cp -r ${src}/datasets ${dst}/datasets
  
  if [ -d "${dst}/save_confidence" ]; then
    rm -rf ${dst}/save_confidence
  fi
  cp -r ${src}/save_confidence ${dst}/save_confidence
  
  cp ${src}/output.log ${dst}/output.log
  
else
  echo "${src} does not exist."
  exit
fi