GBK_ROOT = data/gbk/

Pipfile.lock :
	pipenv install

config.yml : Pipfile.lock
	pipenv run python src/initialize.py

data/json/taxon.json :
	pipenv run python src/fetch_gbk_and_taxon.py --config config.yml

data/json/acc2taxon.json : data/json/taxon.json
	pipenv run python src/filtering.py data/json/taxon.json --config config.yml --filter --out data/json/acc2taxon.json --verbose

data/json/weights.json : data/json/acc2taxon.json
	pipenv run python src/generate_weights.py --json data/json/acc2taxon.json --n-split 3 --out data/json/weights.json --gbk-root $(GBK_ROOT) --verbose

data/coordinates : data/json/weights.json
	pipenv run python src/calculate_coordinates.py data/json/acc2taxon.json --config config.yml --weights data/json/weights.json --gbk-root $(GBK_ROOT) --out data/coordinates/ --verbose

data/img : data/coordinates
	pipenv run python src/generate_img_3d.py data/coordinates/*.json --config config.yml --gradation --out data/img/ --verbose

data/img_gravity : data/coordinates
	pipenv run python src/generate_img_3d.py data/coordinates/*.json --config config.yml --gradation --out data/img_gravity/ --move-gravity --verbose
