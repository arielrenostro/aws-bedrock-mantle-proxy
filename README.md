# Bedrock Mantle Reverse Proxy

Proxy reverso que expõe uma API compatível com **Anthropic** (`/v1/messages`) e uma compatível com **OpenAI** (`/v1/chat/completions`, `/v1/models`), redirecionando as requisições para o **Amazon Bedrock Mantle** por trás.

Isso permite plugar ferramentas como **Claude Code**, **QwenCode** ou **Pi Harness** apontando-as para este proxy local — elas continuam "achando" que estão falando com a Anthropic ou com a OpenAI, mas as requisições são autenticadas e encaminhadas para o Mantle.

## Por que isso existe

O Mantle não permite gerar uma chave de API estática por requisitos de segurança. O acesso precisa ser feito via SDK da AWS, gerando um **bearer token de curta duração** a cada chamada com a biblioteca [`aws-bedrock-token-generator`](https://pypi.org/project/aws-bedrock-token-generator/), a partir das credenciais AWS já configuradas na máquina (perfil, SSO, role assumida, variáveis de ambiente etc.).

O Mantle já expõe nativamente uma API compatível com OpenAI em `https://bedrock-mantle.{region}.api.aws/v1`. Este proxy:

1. Repassa quase que diretamente chamadas no formato OpenAI (injetando o token em cada requisição).
2. Traduz chamadas no formato Anthropic Messages API para o formato OpenAI antes de enviar ao Mantle, e traduz a resposta de volta — inclusive em streaming (SSE).

## Arquitetura

```
app/
  config.py                       # variáveis de ambiente / configuração
  auth.py                         # gera um token Mantle novo a cada requisição
  main.py                         # app FastAPI, registra os routers
  routers/
    anthropic_router.py           # POST /v1/messages  (traduz Anthropic <-> OpenAI)
    openai_router.py              # GET /v1/models, POST /v1/chat/completions (passthrough)
  translation/
    anthropic_to_openai.py        # request: Anthropic Messages -> OpenAI Chat Completions
    openai_to_anthropic.py        # response: OpenAI (JSON e streaming SSE) -> Anthropic Messages
tests/
  test_translation.py             # testes unitários da tradução (sem rede/AWS)
main.py                           # entrypoint (uvicorn)
```

### Fluxo

- **Ferramentas OpenAI-compatible** (QwenCode, Pi Harness, SDK OpenAI genérico) → `POST /v1/chat/completions` ou `GET /v1/models` → proxy injeta o Bearer token → encaminha para o Mantle sem alterar o corpo.
- **Claude Code** (fala o formato Anthropic) → `POST /v1/messages` → proxy traduz a requisição para o formato OpenAI, chama o Mantle, traduz a resposta (ou o stream) de volta para o formato Anthropic.

O nome do modelo (`model`) é repassado como o cliente enviar — **não há mapeamento de nomes amigáveis**. Use `GET /v1/models` para listar os IDs reais disponíveis no Bedrock e aponte suas ferramentas diretamente para eles.

## Pré-requisitos

- Python 3.11+
- Credenciais AWS configuradas localmente com acesso ao Bedrock Mantle (perfil em `~/.aws/credentials`, SSO, variáveis de ambiente ou role assumida) — a resolução usa a cadeia padrão do boto3.

## Instalação

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows
# source .venv/bin/activate # Linux/macOS

pip install -r requirements.txt
```

## Configuração

Copie o arquivo de exemplo e ajuste conforme necessário:

```bash
cp .env.example .env
```

| Variável | Padrão | Descrição |
|---|---|---|
| `AWS_REGION` | `us-east-1` | Região onde o Bedrock Mantle está disponível |
| `MANTLE_BASE_URL` | `https://bedrock-mantle.{AWS_REGION}.api.aws/v1` | Override do endpoint do Mantle |
| `BEDROCK_TOKEN_TTL_SECONDS` | `3600` | Duração solicitada para cada token gerado (máx. 12h / 43200s) |
| `MANTLE_REQUEST_TIMEOUT_SECONDS` | `300` | Timeout das requisições encaminhadas ao Mantle |
| `PROXY_HOST` | `0.0.0.0` | Endereço de bind do servidor local |
| `PROXY_PORT` | `8000` | Porta do servidor local |

## Executando

```bash
python main.py
```

O servidor sobe em `http://localhost:8000` (ou na porta configurada). Verifique com:

```bash
curl http://localhost:8000/healthz
```

## Conectando as ferramentas

### Claude Code (Anthropic)

```bash
export ANTHROPIC_BASE_URL=http://localhost:8000
export ANTHROPIC_API_KEY=qualquer-valor   # ignorado pelo proxy; a auth real é feita no Mantle
```

### Ferramentas OpenAI-compatible (QwenCode, Pi Harness, SDK OpenAI)

```bash
export OPENAI_BASE_URL=http://localhost:8000/v1
export OPENAI_API_KEY=qualquer-valor      # ignorado pelo proxy; a auth real é feita no Mantle
```

Liste os modelos disponíveis no Bedrock:

```bash
curl http://localhost:8000/v1/models
```

## Testes

Os testes de tradução são unitários e não fazem chamadas de rede nem exigem credenciais AWS:

```bash
pytest tests/ -v
```

## Limitações conhecidas

- `/v1/messages/count_tokens` (Anthropic) não está implementado.
- Em respostas **streaming** do endpoint `/v1/messages`, o campo `usage` pode vir zerado caso o Mantle não envie um chunk final com informações de uso. Em respostas não-streaming, o `usage` reportado é real.
- Não há mapeamento de nomes amigáveis de modelo — o valor de `model` é repassado como veio da ferramenta cliente.
