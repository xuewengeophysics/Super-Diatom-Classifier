ipynb_files=$(find . -name \*.ipynb)

for file in $ipynb_files
do
	jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "$file"   
done
