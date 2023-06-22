library(readr)
library(readxl)
#synthetic_train <- read_csv("C:/Users/ru84fuj/Desktop/results/3/synthetic_train.csv")
benchmark <- read_excel("C:/Users/ru84fuj/Desktop/results/benchmark_own_vision_tesseract_40000.xlsx")
#benchmark <- read_excel("C:/Users/ru84fuj/Desktop/results/cloud_latin_bench.xlsx")


benchmark$Model <- factor(benchmark$Model, levels=c("EasyOCR", "Tesseract OCR", "PaddleOCR", "Google Cloud Vision", "TrOCR (fine-tuned)", "ViT+GPT2", "ViT+pretrained GPT2", "Ours", "Ours + pre-trained BERT"))
benchmark$CER <- round(benchmark$CER, 5)


# Focus on test set only
benchmark<-benchmark[benchmark$Split == "Test",]

# Exclude GPT2
benchmark <-benchmark[benchmark$Model != "ViT+GPT2" & benchmark$Model != "ViT+pretrained GPT2" & benchmark$Model != "Ours + pre-trained BERT" , ]

library(ggplot2)
library(dplyr)
library(scales)
library(grid)
library(gridExtra)
library(ggthemes)

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


#colnames(summary_df)<-c("Dataset", "Number of examples", "Mean", "Median", "Min", "Max", "Standard Deviation")

cer_plot <- ggplot(benchmark, aes(x = Model, y = CER, fill = Model)) +
  geom_violin(width = 1) +
  geom_boxplot(width = 0.05, color = "black", alpha = 0.2) +
  xlab("Model") +
  ylab("CER") +
  ylim(c(0,1)) +
  labs(title = "Model benchmarking", 
       subtitle = paste0("CER over ", nrow(benchmark) / length(unique(benchmark$Model)), " test examples"),
       #caption = "*Real data include inaccurate labels (10%) - relabeling ongoing"
  ) +
  theme_economist() +
  guides(fill = "none") +
  scale_fill_economist() +
  theme(
    axis.title = element_text(face = "bold", size = 14)
  )


density_plot<-ggplot(benchmark, aes(x = CER, fill = Model))+geom_density(alpha = 0.50)+
  xlab("CER")+ylab("Frequency")+xlim(c(0,1))+labs(title = "Model benchmarking", 
                                                  subtitle = paste0("CER over ", nrow(benchmark)/length(unique(benchmark$Model)), " test examples"))+theme_economist() + scale_fill_economist()



caption = "*All predictions were performed over real images"



tt <- ttheme_default(colhead=list(fg_params = list(parse=TRUE)),
                     base_size = 12,
                     padding = unit(c(20, 4), "mm"))

tbl <- tableGrob(summary_df, rows=NULL, theme=tt)

grid.arrange(cer_plot, tbl, 
             nrow = 2, heights = c(4, 2))

grid.arrange(density_plot, tbl, 
             nrow = 2, heights = c(4, 2))

