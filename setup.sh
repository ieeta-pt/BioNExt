# first
# create a python environment
echo "Creating Python environment"
python -m venv venv
PIP=venv/bin/pip
$PIP install --upgrade pip
$PIP install -r requirements.txt

# second
# download the embeddings from zenodo
echo "Downloading embeddings..."

cd knowledge-bases
echo "Downloading Cellosaurus embeddings..."
wget -O Cellosaurus.zip https://zenodo.org/records/11126786/files/Cellosaurus.zip?download=1
echo "Extracting Cellosaurus..."
unzip Cellosaurus.zip
rm Cellosaurus.zip

echo "Downloading CTD-diseases embeddings..."
wget -O CTD-diseases.zip https://zenodo.org/records/11126786/files/CTD-diseases.zip?download=1
echo "Extracting CTD-diseases..."
unzip CTD-diseases.zip
rm CTD-diseases.zip

echo "Downloading MeSH embeddings..."
wget -O MeSH.zip https://zenodo.org/records/11126786/files/MeSH.zip?download=1
echo "Extracting MeSH..."
unzip MeSH.zip
rm MeSH.zip

echo "Downloading NCBI-Gene embeddings..."
wget -O NCBI-Gene.zip https://zenodo.org/records/11126786/files/NCBI-Gene.zip?download=1
echo "Extracting NCBI-Gene..."
unzip NCBI-Gene.zip
rm NCBI-Gene.zip

echo "Downloading NCBI-Taxonomy embeddings..."
wget -O NCBI-Taxonomy.zip https://zenodo.org/records/11126786/files/NCBI-Taxonomy.zip?download=1
echo "Extracting NCBI-Taxonomy..."
unzip NCBI-Taxonomy.zip
rm NCBI-Taxonomy.zip

cd ..