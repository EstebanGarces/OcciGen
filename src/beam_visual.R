library(readr)
library(readxl)
#synthetic_train <- read_csv("C:/Users/ru84fuj/Desktop/results/3/synthetic_train.csv")

beam_analysis <- read_excel("C:/Users/ru84fuj/Desktop/results/beam_analysis.xlsx")
beam_analysis <- read_excel("C:/Users/ru84fuj/Desktop/help_files/beam_analysis_swin_bert.xlsx")

beam_analysis$Width <- factor(beam_analysis$Width, levels=sort(unique(beam_analysis$Width)))

# Focus on test set only
beam_analysis<-beam_analysis[beam_analysis$Split == "Test",]

library(ggplot2)
library(dplyr)
library(scales)
library(grid)
library(gridExtra)
library(ggthemes)

summary_df <- beam_analysis %>% group_by(Width) %>% summarize(Mean = round(mean(CER),5),
                                                              MeanWeighted = round(sum(Weighted_CER)/sum(Length), 5),
                                                              Median = round(median(CER),5),
                                                              Min = round(min(CER),2),
                                                              Max = round(max(CER),2),
                                                              StdDev = round(sd(CER),4),
                                                              'Correctly predicted labels (%)' = round(mean(Correct), 3)*100) %>%
  as.data.frame()
                                                              

cer_plot <- ggplot(beam_analysis, aes(x = Width, y = CER, fill = Width)) +
  geom_violin(width = 0.5) +
  geom_boxplot(width = 0.5, color = "black", alpha = 0.2) +
  xlab("Beam width") +
  ylab("CER") +
  ylim(c(0,1)) +
  labs(title = "Model performance by beam width", 
       subtitle = paste0("CER over ", nrow(beam_analysis) / length(unique(beam_analysis$Width)), " test examples")
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

grid.arrange(density_plot, tbl, 
             nrow = 2, heights = c(4, 2))

