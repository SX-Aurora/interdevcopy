#!/bin/bash
cp -f ../README.md page_readme.md
sed -i '/Getting Started with Interdevcopy/s/$/ # {#page_readme}/' page_readme.md
doxygen Doxyfile || exit

# exclude private page
find ./build/html/ -type f -name "classinterdevcopy*.html" -a -not -name "*DeviceMemoryRegion*" -a -not -name "*CopyChannel*" | xargs rm
find ./build/html/ -type f -name "structinterdevcopy_1_1*.html" | xargs rm
find ./build/html/ -type f -name "namespace*.html" | xargs rm
find ./build/html/ -type f -name "dir_*.html" | xargs rm
