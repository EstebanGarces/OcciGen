library(readr)
library(readxl)
library(ggplot2)
library(dplyr)
library(scales)
library(grid)
library(gridExtra)
library(ggthemes)

benchmark <- read_excel("path/to/Occigen/benchmarking/benchmarking_results.xlsx", 
           sheet = "benchmark")

benchmark$Model <- factor(benchmark$Model, levels=c("EasyOCR", "Tesseract OCR", "PaddleOCR", "Google Cloud Vision", "TrOCR (fine-tuned)", "Ours"))
benchmark$CER <- round(benchmark$CER, 5)


# Architectures

summary_df <- benchmark %>% group_by(Model) %>% summarize(Architecture = unique(Architecture),
                                                           Mean = round(mean(CER),3),
                                                           MeanWeighted = round(sum(Weighted_CER)/sum(Length), 3),
                                                           Median = round(median(CER),3),
                                                           Min = round(min(CER),3),
                                                           Max = round(max(CER),3),
                                                           StdDev = round(sd(CER),3),
                                                           'Correctly predicted labels (%)' = round(mean(Correct),3)*100) %>%
  as.data.frame()


cer_plot <- ggplot(benchmark, aes(x = Model, y = CER, fill = Model)) +
  geom_violin(width = 1) +
  geom_boxplot(width = 0.05, color = "black", alpha = 0.2) +
  xlab("Model") +
  ylab("CER") +
  ylim(c(0,1)) +
  labs(title = "Model benchmarking", 
       subtitle = paste0("CER over ", nrow(benchmark) / length(unique(benchmark$Model)), " test examples"),
  ) +
  theme_economist() +
  guides(fill = "none") +
  scale_fill_economist() +
  theme(
    axis.title = element_text(face = "bold", size = 14)
  )

tt <- ttheme_default(colhead=list(fg_params = list(parse=TRUE)),
                     base_size = 12,
                     padding = unit(c(20, 4), "mm"))

tbl <- tableGrob(summary_df, rows=NULL, theme=tt)

grid.arrange(cer_plot, tbl, 
             nrow = 2, heights = c(4, 2))


# Benchmarking with other Occitan data sets

benchmark <- read_excel("path/to/Occigen/benchmarking/benchmarking_results.xlsx", 
                        sheet = "occitan_datasets")
benchmark$`Data set` <- factor(benchmark$`Data set`, levels=c("Blue (DOM Project)", "Red", "Yellow", "Green"))
benchmark$CER <- round(benchmark$CER, 5)

summary_df <- benchmark %>% group_by(`Data set`) %>% summarize(Examples = n(),
                                                               Mean = round(mean(CER),3),
                                                               MeanWeighted = round(sum(Weighted_CER)/sum(Length), 3),
                                                               Median = round(median(CER),3),
                                                               Min = round(min(CER),3),
                                                               Max = round(max(CER),3),
                                                               StdDev = round(sd(CER),3),
                                                               'Correctly predicted labels (%)' = round(mean(Correct),3)*100) %>%
  as.data.frame()


cer_plot <- ggplot(benchmark, aes(x = `Data set`, y = CER, fill = `Data set`)) +
  geom_violin(width = 1) +
  geom_boxplot(width = 0.05, color = "black", alpha = 0.2) +
  xlab("Data set") +
  ylab("CER") +
  ylim(c(0,1)) +
  labs(title = "Model performance on different Old Occitan data sets", 
       subtitle = paste0("CER over ", nrow(benchmark), " test examples"),
  ) +
  theme_economist() +
  guides(fill = "none") +
  scale_fill_manual(values = c("Blue (DOM Project)" = "blue", "Yellow" = "#FFC107", "Red" = "#B71C1C", "Green" = "#1B5E20")) +
  theme(
    axis.title = element_text(face = "bold", size = 14)
  )


tt <- ttheme_default(colhead=list(fg_params = list(parse=TRUE)),
                     base_size = 12,
                     padding = unit(c(20, 4), "mm"))

tbl <- tableGrob(summary_df, rows=NULL, theme=tt)

grid.arrange(cer_plot, tbl, 
             nrow = 2, heights = c(4, 2))




