library(readxl)
library(mirt)
library(psych)

# Selecione a planilha de respostas
arquivo <- file.choose()
dados <- read_excel(arquivo, sheet = "Planilha1", col_names = FALSE)

# Matriz com os 20 itens
respostas <- as.data.frame(dados[, 1:20])
respostas[] <- lapply(respostas, function(x) suppressWarnings(as.numeric(x)))
linhas <- apply(respostas, 1, function(x) all(!is.na(x) & x %in% c(0, 1)))
respostas <- respostas[linhas, ]
names(respostas) <- paste0("Q", 1:20)

dim(respostas)

# Modelo 2PL
modelo <- mirt(respostas, 1, itemtype = "2PL", method = "EM", verbose = FALSE)
coeficientes <- coef(modelo, IRTpars = TRUE, simplify = TRUE)$items

parametros <- data.frame(
  Questao = paste("Questão", 1:20),
  Discriminacao_a = coeficientes[, "a"],
  Dificuldade_b = coeficientes[, "b"]
)

print(parametros)

write.csv2(
  parametros,
  file = file.path(path.expand("~/Downloads"), "parametros_2PL_2024.csv"),
  row.names = FALSE
)

# Alfa de Cronbach
alfa <- psych::alpha(respostas, check.keys = FALSE)$total$raw_alpha
cat("Alfa de Cronbach:", alfa, "\n")

# Curvas dos itens em um único PDF, uma questão por página
theta <- seq(-4, 6.5, length.out = 1000)
saida <- file.path(path.expand("~/Downloads"), "curvas_2PL_2024.pdf")

pdf(saida, width = 8, height = 6, onefile = TRUE)

for (i in 1:20) {
  a <- parametros$Discriminacao_a[i]
  b <- parametros$Dificuldade_b[i]
  p <- 1 / (1 + exp(-a * (theta - b)))

  plot(
    theta, p,
    type = "l",
    lwd = 2,
    col = "blue",
    ylim = c(0, 1),
    xlab = expression(theta),
    ylab = "Probabilidade de acerto",
    main = paste("Curva Característica do Item", i),
    sub = paste0("a = ", round(a, 4), "   b = ", round(b, 4))
  )

  abline(h = 0.5, lty = 2, col = "gray50")
  abline(v = b, lty = 2, col = "red")
  points(b, 0.5, pch = 19, col = "red")
}

dev.off()
cat("PDF salvo em:", saida, "\n")
