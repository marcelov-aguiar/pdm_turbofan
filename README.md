# Criação do ambiente
Antes de criar  ambiente virtual, recomenda-se instalar a versão 3.8.9 do Python. Recomenda-se instalar o Visual Studio Code (VS Code). Após essas instalações, o ambiente virtual poderá ser criado.


## Visual Studio Code
Editor utilizado foi VS Code versão 1.71.0.

### Extensões
- autoDocstring - Python Docstring Generator: Formatação do cabeçalho da função no formato numpy. 
- Jupyter: Permite execução do Jupyter Notebook no VS Code.
- Jupyter Notebook Renderers: Facilita plot de figuras no Jupyter notebook.
- Python: Permite executar scripts Python rapidamente pelo botão de execução.
- Pylance: Verifica em tempo real erros de sintaxe.
- Dev Containers: Permite executar códigos dentro de um container Docker.


## Virtualenv
Recomenda-se criar o ambiente virtual python para executar o código desse projeto. Para isso será necessário instalar o pacote virtualenv versão 20.16.5 conforme o comando abaixo
```
pip install virtualenv==20.16.5
```

Após isso, as bibliotecas listadas em `docker\requirements.txt` podem ser instaladas, conforme o comando abaixo:
```
pip install -r requeriments.txt
```

## Docker
O código implementado porderá ser executado em um container Docker. Para isso os seguinte comando devem ser utilizado:
- Criar o container
```
docker compose build
```

- Após a criação ele poderá ser iniciado por
```
docker compose up -d <container_name>
```

Parar a execução do container o seguinte comando poderá ser executado:
```
docker compose kill deploy_uo2_dp
```

## Padrão de projeto
Foi utilizado um padrão de projeto numpy para documentação do cabeçalho das funções. Para escrita do código foram uilizados os padrões flake8 e autopep8 para fazer a formatação automática. As versões desses pacotes podem ser encontradas em `docker\requirements.txt`. Essa formatação pode ser executada pelo código:
```
autopep8 --in-place --aggressive --aggressive <filename>
```

# Modelagem
## MLFlow
Necessário alterar a constante `MAX_PARAM_VAL_LENGTH` para o valor igual a 1000. Essa alteração permite salvar uma string maior nos parâmetros do MLFlow. A constante se localiza no seguinte arquivo: \Lib\site-packages\mlflow\utils\validation.py


