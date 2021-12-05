# MtGenome ml classification

## setup

- Sync packages
    ```sh
    make Pipfile.lock
    ```
- Generate `config.yml`
    ```sh
    make config.yml
    ```
- Write `config.yml`

- Fetch genbank file and taxonomy information
    ```sh
    make data/json/taxon.json
    ```

- Filtering
    ```sh
    make data/json/acc2taxon.json
    ```

- Calculate coordinates
    ```sh
    make data/coordinates
    ```

- Generate images
    ```sh
    make data/img
    ```

- Machine learning
    ```sh
    pipenv run leaning --description <DESCRIPTION>
    ```
