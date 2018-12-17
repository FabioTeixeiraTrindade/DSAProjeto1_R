# Título: Detecção de Fraudes no Tráfego de Cliques 
# em Propagandas de Aplicações Mobile utilizando Random Forest,
# Árvore de Decisão, SVM, Regressão Logística e Naive Bayes.
# Autor: Fábio Teixeira Trindade
# Data: 03/Dez/2018

# Introdução
  
# Este é um problema de Classificação

# A maior e independente plataforma de serviço de dados de big data da china
# desafiou-nos a construir um algoritmo para predizer se um usuário
# vai baixar um app depois de clicar numa propaganda num celular.

# E os dados disponíveis a nós são dados mascarados (Somente podemos ver os códigos,
# não os dados reais)

# Esse é uma das práticas que todo mundo precisa adotar como iniciativa GDPR (Regulação 
# de Proteção de dados Genéricos), isso deve ser adotado por todas as companhias que está nossos dados

# Para informar sobre GDPR, GDPR terá efeito no dia 25 de maio de 2018 para todos os paises
# que processam dados europeus e paises que não terão conformidade com GDPR
# terão penalidades com relação a ele.

# Cada linha dos dados de treino contém o registro de um click, com as seguintes variáveis:
#
#ip: endereço ip do click
#app: identificação do app para usada no marketing
#device: identificação do tipo de dispositivo do celular do usuário (por exempolo, iphone 6 plus, iphone 7, huawei mate 7, etc.)
#os: versão do sistema operacional do celular
#channel: canal de quem faz a chamado do marketing no celular
#click_time: data e hora do click (UTC)
#attributed_time: Se o usuário fizer o download após clicar na propaganda, essa é a hora de download do app
#is_attributed: variável target que será predita no modelo, que indica se a app foi baixada (downloaded)

# mostra o diretório de trabalho
getwd()
# Desliga mensagens que só atrapalham
options(warn=-1)

# Carrega todas as bibliotecas necessárias
library(lubridate)
library(caret)
library(dplyr)
library(DMwR)
library(ROSE)
library(ggplot2)
library(randomForest)
library(rpart)
library(rpart.plot)
library(data.table)
library(e1071)
library(gridExtra)

# Lê os dados de treino e de teste e verifica sua estrutura
train <-fread('train.csv', stringsAsFactors = FALSE, data.table = FALSE)
test <-fread('test.csv', stringsAsFactors = FALSE, data.table = FALSE)

str(train)
str(test)

# Não há diferença entre dados de treino e teste, a não ser pela presença da variável
# target (is_attributed) nos dados de teste que devemos prever 
# e da variável attributed_time (tempo levado para o download do app)
# que vem como NA nos dados de teste)
# verificando e estimando valores missing
colSums(is.na(train))

# Não há valores missing, os dados estão limpos
# Attributes_time (Tempo levado para download) tem valores em branco,
# isso está logicamente correto
colSums(train=='')

## Vamos verificar a variável target, quantos não foram baixados no dados de treino
table(train$is_attributed)

# Verifica-se que a suposição está correta na medida em que comparamos
# os valores em branco em Attributes_time e a quantidade de aplicações não baixadas
# (is_attributed = 0) nos dados de treino.
# Como esses valores batem, não precisa-se fazer mais nada.
# Observa-se que essa variável não está presente nos dados de teste,
# portanto, não há motivos para mantê-la nos dados do treino também

train$attributed_time=NULL

# Data Munging,Análise Exploratória de Dados, e Feature Engineering
# Feature Engineering

# Feature engineering é o processo de determinar quais variáveis preditoras
# contribuirão para a capacidade preditiva do algoritmo de aprendizado de máquina. 
# Engenharia de Atributos (Feature engineering)
# Análise Exploratória de Dados

# Como a maioria das variáveis estão mascaradas, 
# não tem muito o que fazer em feature engineering
# Contudo, podemos extrair algumas variáveis de click_time, variável tipo data e hora,
# para fazer algumas análises desses dados

# Convertendo click_time para o formato data e hora
train$click_time<-as.POSIXct(train$click_time,
                             format = "%Y-%m-%d %H:%M",tz = "America/New_York")

# separando ano, mês, dia da semana e hora
train$year=year(train$click_time)
train$month=month(train$click_time)
train$days=weekdays(train$click_time)
train$hour=hour(train$click_time)

# Depois de obter essas novas variáveis, vamos remover a variável original "click_time"
train$click_time=NULL

# Verificando valores únicos para cada variável obtida de click_time
apply(train,2, function(x) length(unique(x)))

# Olhando os valores únicos, podemos ver que as datas coletadas para um único mês,
# em um determinado ano, ou seja, o dados se repete a cada linha e então,
# não há necessidade de manter essas variáveis mês e ano 

train$month=NULL
train$year=NULL

# Convertendo as variáveis "is_attributed" e "days" em variáveis do tipo fator

train$is_attributed=as.factor(train$is_attributed)
train$days=as.factor(train$days)

# Análise exploratória de dados para verificar a importância das variáveis para
# a previsão

# App foi baixado (is_attributed) versus App_id para marketing

p1=ggplot(train,aes(x=is_attributed,y=app,fill=is_attributed))+
  geom_boxplot()+
  ggtitle("Application ID versus Is_attributed")+
  xlab("App ID") +
  labs(fill = "is_attributed")  

p2=ggplot(train,aes(x=app,fill=is_attributed))+
  geom_density()+facet_grid(is_attributed~.)+
  scale_x_continuous(breaks = c(0,50,100,200,300,400))+
  ggtitle("Application ID versus Is_attributed")+
  xlab("App ID") +
  labs(fill = "is_attributed")  

p3=ggplot(train,aes(x=is_attributed,y=app,fill=is_attributed))+
  geom_violin()+
  ggtitle("Application ID versus Is_attributed")+
  xlab("App ID") +
  labs(fill = "is_attributed")  

grid.arrange(p1,p2, p3, nrow=2,ncol=2)

# Observe o padrão e a forma diferente em todos os gráficos App foi baixado versus
# App id no marketing, especialmente a diferenciação clara no Boxplot
# Isso definitivamente vai ser umas das variáveis importantes para diferenciar
# usuários que baixaram a aplicação ou não

# App foi baixada versus versão do sistema operacional (OS) do celular
p4=ggplot(train,aes(x=is_attributed,y=os,fill=is_attributed))+
  geom_boxplot()+
  ggtitle("Os version versus Is_attributed")+
  xlab("OS version") +
  labs(fill = "is_attributed")  


p5=ggplot(train,aes(x=os,fill=is_attributed))+
  geom_density()+facet_grid(is_attributed~.)+
  scale_x_continuous(breaks = c(0,50,100,200,300,400))+
  ggtitle("Os version versus Is_attributed ")+
  xlab("Os version") +
  labs(fill = "is_attributed")


p6=ggplot(train,aes(x=is_attributed,y=os,fill=is_attributed))+
  geom_violin()+
  ggtitle("Os version versus Is_attributed")+
  xlab("Os version") +
  labs(fill = "is_attributed")  


grid.arrange(p4,p5, p6, nrow=2,ncol=2)

# Não tem diferenciação entre as duas variáveis. Assim a versão do sistema 
# operacional (os) não é uma variável importante para a previsão


# App foi baixada versus endereço ip do click.
p7=ggplot(train,aes(x=is_attributed,y=ip,fill=is_attributed))+
  geom_boxplot()+
  ggtitle("IP Address versus Is_attributed")+
  xlab("Ip Adresss of click") +
  labs(fill = "is_attributed")  


p8=ggplot(train,aes(x=ip,fill=is_attributed))+
  geom_density()+facet_grid(is_attributed~.)+
  scale_x_continuous(breaks = c(0,50,100,200,300,400))+
  ggtitle("IP Address versus Is_attributed")+
  xlab("Ip Adresss of click") +
  labs(fill = "is_attributed")  



p9=ggplot(train,aes(x=is_attributed,y=ip,fill=is_attributed))+
  geom_violin()+
  ggtitle("IP Address versus Is_attributed")+
  xlab("Ip Adresss of click") +
  labs(fill = "is_attributed")  

grid.arrange(p7,p8, p9, nrow=2,ncol=2)


# O endereço IP (IP Address) pode, muito bem, desempenhar um papel importante
# na previsão pois há diferenciação entre os dois grupos


# App foi baixada versus ID do tipo de dispositivo do usuário

p10=ggplot(train,aes(x=device,fill=is_attributed))+
  geom_density()+facet_grid(is_attributed~.)+
  ggtitle("Device type versus Is_attributed")+
  xlab("Device Type ID") +
  labs(fill = "is_attributed")  


p11=ggplot(train,aes(x=is_attributed,y=device,fill=is_attributed))+
  geom_boxplot()+
  ggtitle("Device type versus Is_attributed")+
  xlab("Device Type ID") +
  labs(fill = "is_attributed")  


p12=ggplot(train,aes(x=is_attributed,y=device,fill=is_attributed))+
  geom_violin()+
  ggtitle("Device type versus Is_attributed")+
  xlab("Device Type ID") +
  labs(fill = "is_attributed")  

grid.arrange(p10,p11, p12, nrow=2,ncol=2)

# Não há diferenciação entre dispositivo (device) e is_attributed,
# não sendo importante para nossa análise

#App foi baixada (is_attributed) versus ID do canal do editor de anúncios para celular

p13=ggplot(train,aes(x=channel,fill=is_attributed))+
  geom_density()+facet_grid(is_attributed~.)+
  ggtitle("Channel versus Is_attributed")+
  xlab("Channel of mobile") +
  labs(fill = "is_attributed")  


p14=ggplot(train,aes(x=is_attributed,y=channel,fill=is_attributed))+
  geom_boxplot()+
  ggtitle("Channel versus Is_attributed")+
  xlab("Channel of mobile") +
  labs(fill = "is_attributed")  

p15=ggplot(train,aes(x=is_attributed,y=channel,fill=is_attributed))+
  geom_violin()+
  ggtitle("Channel versus Is_attributed")+
  xlab("Channel of mobile") +
  labs(fill = "is_attributed")  

grid.arrange(p13,p14, p15, nrow=2,ncol=2)

# O canal do editor tem possibilidades de ajudar na previsão,
# podemos usar essa variável na análise de variáveis (feature)

# A hora específica tem alguma relação com o download do app 

p16=ggplot(train,aes(x=hour,fill=is_attributed))+
  geom_density()+facet_grid(is_attributed~.)+
  ggtitle("Hour versus Is_attributed ")+
  xlab("Hour") +
  labs(fill = "is_attributed")  

p17=ggplot(train,aes(x=is_attributed,y=hour,fill=is_attributed))+
  geom_boxplot()+
  ggtitle("Hour versus Is_attributed")+
  xlab("Hour") +
  labs(fill = "is_attributed")  

p18=ggplot(train,aes(x=is_attributed,y=channel,fill=is_attributed))+
  geom_violin()+
  ggtitle("Hour versus Is_attributed")+
  xlab("Hour") +
  labs(fill = "is_attributed")  

grid.arrange(p16,p17, p18, nrow=2,ncol=2)


# Há uma leve diferenciação em ambas as distruibuição,
# podemos dizer que é uma variável menos importante

# Um dia específico tem algo a ver com o download da aplicação?

p19=ggplot(train,aes(x=days,fill=is_attributed))+
  geom_density()+facet_grid(is_attributed~.)+
  ggtitle("Dia da semana versus Is_attributed ")+
  xlab("Os version") +
  labs(fill = "is_attributed")  


p20=ggplot(train,aes(x=days,fill=is_attributed))+geom_density(col=NA,alpha=0.35)+
  ggtitle("dias versus clique")+
  xlab("Dia da semana versus Is_attributed ") +
  ylab("Total Count") +
  labs(fill = "is_attributed")  

grid.arrange(p19,p20, ncol=2)

# Parece que não há relação entre a variável dia e attributed_id

# Validação sobre os insights das variáveis

# Vamos randomicamente aplicar algum modelo
# 1) para todas as variáveis
# 2) para variáveis selecionadas por meio da análise exploratória de dados


# 1) Modelo para todas as variáveis

# Utilizando o pacote caret para particionar os dados de treino para aplicar ao modelo
set.seed(1234)
cv.10 <- createMultiFolds(train$is_attributed, k = 10, times = 10)

# Utilização da validação cruzada (cross-validation) que divide os dados em 10
# partes e roda o modelo 10 vezes, cada vez usando uma das partes diferentes como validação.
# O método repeatedcv é um bom começo.
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                     index = cv.10)


# Esta função configura conjuntos de dados de treino para uma série de classificações e regressões,
# ajusta os modelos e calcula uma medida de desempenho baseada em reamostragem (partições dos dados de treino).
set.seed(1234)
Model_CDT <- train(x = train[,-6], y = train[,6], method = "rpart", tuneLength = 30,
                   trControl = ctrl)

PRE_VDTS=predict(Model_CDT$finalModel,data=train,type="class")
confusionMatrix(PRE_VDTS,train$is_attributed)

## Verificando a acurácia, apesar da acurácia está muito alta, mas a especificidade é muito baixa

# 2 ) Modelo para variáveis selecionadas
train$days=NULL
train$os=NULL
train$device=NULL

set.seed(1234)

Model_CDT1 <- train(x = train[,-4], y = train[,4], method = "rpart", tuneLength = 30,
                    trControl = ctrl)

PRE_VDTS1=predict(Model_CDT1$finalModel,data=train,type="class")
confusionMatrix(PRE_VDTS1,train$is_attributed)

# Nesse segundo modelo, chega-se a mesma acurácia, no entanto há uma mudança drástica na especificidade
# Então, iniciarie usando somente variáveis selecionadas para o nosso modelo atual
# Particição dos dados
# Antes de fazer qualquer coisa, Vamos dividir os dados em dados de treino e dados de testes usando pacote caret

set.seed(5000)
ind=createDataPartition(train$is_attributed,times=1,p=0.7,list=FALSE)
train_val=train[ind,]
test_val=train[-ind,]

# Verificar a proporção 
round(prop.table(table(train$is_attributed)*100),digits = 3)
round(prop.table(table(train_val$is_attributed)*100),digits = 3)
round(prop.table(table(test_val$is_attributed)*100),digits = 3)

# Observe que o Caret divide os dados na taxa de 70% e 30% 

# Balanceando os dados usando o Smote

# No final, verifiquei todas as técnicas tais como up sampling, down sampling, Rose and smote,
# e entre esses, o Smote se sobressaiu com boa acurária

# Vamos aplicar o smote e tentar equilibrar os dados
set.seed(1234)
smote_train = SMOTE(is_attributed ~ ., data  = train_val)                         
table(smote_train$is_attributed) 

# Algoritmo de Aprendizado de Máquina e Validação Cruzada

# Árvore de Decisão

set.seed(1234)
cv.10 <- createMultiFolds(smote_train$is_attributed, k = 10, times = 10)

# Controle
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                     index = cv.10)
set.seed(1234)
# Treina o modelo
Model_CDT <- train(x = smote_train[,-4], y = smote_train[,4], method = "rpart", tuneLength = 30,
                   trControl = ctrl)

rpart.plot(Model_CDT$finalModel,extra =  3,fallen.leaves = T)


PRE_VDTS=predict(Model_CDT$finalModel,newdata=test_val,type="class")
confusionMatrix(PRE_VDTS,test_val$is_attributed)

# Somos capazes de completar Árvore de Decisão com 0,94% de acurácia,
# e especificidade aumentada para 0,78% (Lembre-se,
# aumento drástico em especificidade depois do balanceamento dos dados)

###Random forest


cv.10 <- createMultiFolds(smote_train$is_attributed, k = 10, times = 10)

# Controle
ctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 10,
                     index = cv.10)
set.seed(1234)
set.seed(1234)
rf.5<- train(x = smote_train[,-4], y = smote_train[,4], method = "rf", tuneLength = 3,
             ntree = 100, trControl =ctrl)

rf.5

pr.rf=predict(rf.5,newdata = test_val)

confusionMatrix(pr.rf,test_val$is_attributed)

##O modelo Random forest nos dá 95% accuracy, 1% melhor que decision tree,
# mas observe que não há muita mudança na especificidade


###Support Vector Machine (SVM)
###Linear Support vector Machine (LSVM)

##Antes de entrar no modelo, vamos ajustar o parâmetro custo (pacote e1071)

set.seed(1234)
liner.tune=tune.svm(is_attributed~.,data=smote_train,kernel="linear",cost=c(0.1,0.5,1,5,10,50))

liner.tune

### Vamos pegar o melhor modelo linear
best.linear=liner.tune$best.model

##Dados de previsão

best.test=predict(best.linear,newdata=test_val,type="class")
confusionMatrix(best.test,test_val$is_attributed)

#A Acurácia diminui no modelo Linear SVM, SVM não é um bom modelo para esses dados
#Radial Support vector Machine

# Vamos aplicar o SVM não linear, Radial Kernel

set.seed(1234)
rd.poly=tune.svm(is_attributed~.,data=smote_train,kernel="radial",gamma=seq(0.1,5))

summary(rd.poly)

best.rd=rd.poly$best.model


##Vamos fazer previsões nos dados de teste
pre.rd=predict(best.rd,newdata = test_val)

confusionMatrix(pre.rd,test_val$is_attributed)
##Embora o kernel Radial faz melhor que o linear, no geral, a precisão não é boa.

#Conclusão: poderíamos ter alcançado 99% de acurácia simplesmente usando os dados sem fazer "class balance".

# REGRESSÃO LOGÍSTICA

# Treinando o modelo
log.model <- glm(formula = is_attributed ~ . , family = binomial(link = 'logit'), data = train)

# Podemos ver que as variaveis Sex, Age e Pclass sao as variaveis mais significantes
summary(log.model)

# Fazendo as previsoes nos dados de teste
library(caTools)
set.seed(101)

# Split dos dados
split = sample.split(train$is_attributed, SplitRatio = 0.70)

# Datasets de treino e de teste
dados_treino_final = subset(train, split == TRUE)
dados_teste_final = subset(test, split == FALSE)

# Gerando o modelo com a versao final do dataset
final.log.model <- glm(formula = is_attributed ~ . , family = binomial(link='logit'), data = dados_treino_final)

# Resumo
summary(final.log.model)
library(Amelia)
# Prevendo a acuracia
fitted.probabilities <- predict(final.log.model, newdata = dados_treino_final, type = 'response')

# Calculando os valores
fitted.results <- ifelse(fitted.probabilities > 0.5, 1, 0)

# Conseguimos 99% de acuracia
misClasificError <- mean(fitted.results != dados_treino_final$is_attributed)
print(paste('Acuracia', (1-misClasificError)*100))

# Criando a confusion matrix
table(dados_treino_final$is_attributed, fitted.probabilities > 0.5)

confusionMatrix(factor(fitted.results),dados_treino_final$is_attributed)

# Criando o modelo NAIVE BAYES
nb_model <- naiveBayes(is_attributed ~ ., data = train)

# Visualizando o resultado
nb_model
summary(nb_model)
str(nb_model)

# Previsoes
nb_test_predict <- predict(nb_model, train)

# Confusion matrix
confusionMatrix(nb_test_predict,train$is_attributed)

