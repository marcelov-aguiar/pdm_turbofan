# Criação do ambiente
Antes de criar  ambiente virtual, recomenda-se instalar a versão 3.8.9 do Python. Após essa instalação, o ambiente virtual poderá ser instalado.

## Virtualenv
Recomenda-se criar o ambiente virtual python para executar o código desse projeto. Para isso será necessário instalar o pacote virtualenv versão 20.16.5 conforme o comando abaixo
```
pip install virtualenv==20.16.5
```

Após isso, as bibliotecas listadas em `docker\requirements.txt` podem ser instaladas, conforme o comando abaixo:
```
pip install -r requeriments.txt
```