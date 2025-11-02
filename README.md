# Trabalho Sistemas Colaborativos
## SSC0723 - Sistemas Colaborativos: Fundamentos e Aplicações (2025)
Laura Ferré Scotelari - 12543436

### Descrição do Cenário
Após a entrada de novos membros, uma equipe de desenvolvedores precisa revisar e sistematizar as Definitions of Done (DoD) e Definitions of Ready (DoR) do time. Para isso, eles decidiram analisar as anotações já existentes em um PDF e utilizar uma LLM para auxiliar na tarefa, conciliando tanto as especificidades da equipe quanto padrões amplamente adotados na área.

**Participantes (Devs):** Pedro, Arthur, Sarah, Gustavo.

**Objetivo:** Elaborar definições coletivas relacionadas com os DoD e os DoR já existentes nos PDFs, permitindo que múltiplos usuários interajam com o agente LLM.

### Como o sistema aborda cada um dos 3C?

- **Comunicação:** ocorre em um sistema de chat que engloba os usuários (devs) e a LLM, onde cada participante pode participar das trocas de mensagem.
- **Colaboração:** as definições são elaboradas conjuntamente em um documento compartilhado, onde todos têm acesso às informações registradas e podem opinar.
- **Coordenação**: Controle de etapas e fluxo de tarefas, além de um gerenciamento de histórico.

### Diagrama do Grafo LangGraph

O grafo utilizado é composto pelos seguintes nós: 

- **UserInput** → Input do usuário.
- **RAGQuery** → Busca de informações no documento disponibilizado.
- **LLMResponse** → Chama o LLM para gerar uma resposta.
- **ToolNode** → Uso de ferramentas específicas (ex: sumarização, votação, atualização de documento).
- **DecisionNode** → Decide se o fluxo continua (nova pergunta, revisão ou fim).
- **END** → Finaliza o fluxo.
