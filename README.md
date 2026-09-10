# Análise psicométrica com IA e R

Este repositório reúne os códigos, a matriz de respostas anonimizada e as figuras utilizados no estudo **Análise psicométrica com inteligência artificial e R: comparação de estimativas em uma prova de seleção para o 6º ano**.

Foram comparadas duas estimações dos parâmetros de discriminação e dificuldade do modelo logístico de dois parâmetros da Teoria da Resposta ao Item:

- uma implementação própria em Python, elaborada e executada pelo GPT-5.6 Sol no ambiente do ChatGPT;
- uma análise independente no R com o pacote `mirt`.

## Arquivos

- `dados/matriz_respostas_2024_anonimizada.xlsx`: matriz dicotômica com 971 respondentes e 20 itens, sem identificadores pessoais, gabarito ou totais;
- `codigo/analise_tri_python_gpt56sol.py`: código integral efetivamente executado pela IA, preservado sem reformulação;
- `codigo/analise_tri_mirt.R`: código utilizado para reproduzir a análise no R, calcular o alfa de Cronbach e gerar as curvas dos itens em PDF;
- `figuras/`: Curvas Características dos Itens 5 e 10 apresentadas no artigo;
- `requirements-python.txt`: versões das principais dependências do ambiente Python.

## Dados

A análise utilizou respostas dicotômicas de 971 candidatos a 20 itens de uma prova de Matemática aplicada em 2024. Para favorecer a transparência e a reprodução dos cálculos, o repositório disponibiliza uma cópia anonimizada contendo apenas a matriz 0/1 empregada nas análises. Foram removidos o gabarito, os totais, os metadados de autoria e quaisquer informações que pudessem identificar a instituição ou os candidatos.

## Procedimento em Python

O código Python não utiliza pacote psicométrico específico. A estimação foi programada com operações do NumPy e funções numéricas do SciPy. No modelo 2PL, empregaram-se máxima verossimilhança marginal, algoritmo EM, distribuição normal padrão para a proficiência, quadratura Gauss-Hermite com 41 pontos e otimização L-BFGS-B. O código impôs os limites de 0,15 a 4 ao parâmetro de discriminação e de -5 a 5 ao parâmetro de dificuldade.

O arquivo contém caminhos do ambiente original de execução. Para reproduzir a análise com outro banco autorizado, é necessário alterar as variáveis `SRC` e `OUT` no início do script.

## Procedimento no R

O script R abre uma janela para seleção da planilha, organiza a matriz dicotômica, ajusta o modelo 2PL pelo pacote `mirt`, calcula o alfa de Cronbach pelo pacote `psych`, exporta os parâmetros e produz um único PDF com uma curva por página. O arquivo gráfico é criado pelo dispositivo `pdf()` do R e não depende de `ragg` ou XQuartz.

## Observação sobre a autoria do código

A expressão "implementação própria" indica que o script não chamou uma rotina pronta de TRI em Python. Ela não significa que tenha sido demonstrada novidade algorítmica: o programa operacionaliza formulações estatísticas conhecidas da TRI por meio de bibliotecas numéricas gerais. O histórico preservado não permite vincular o código a uma fonte específica nem determinar se padrões aprendidos pelo modelo durante seu treinamento tiveram origem em códigos de Python ou de outras linguagens.
