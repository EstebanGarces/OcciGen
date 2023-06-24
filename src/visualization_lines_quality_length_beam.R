library(readr)
library(readxl)
library(ggplot2)
library(dplyr)
library(scales)
library(grid)
library(gridExtra)
library(ggthemes)

# Read file
benchmark <- read_excel("/path/to/error_analysis_quality_lines_length.xlsx")

# Error analysis based on number of lines
benchmark$Lines <- factor(benchmark$Lines, levels=unique(benchmark$Lines))
benchmark$CER <- round(benchmark$CER, 4)


summary_df <- benchmark %>% group_by(Lines) %>% summarize(Examples = n(),
                                                          Mean = round(mean(CER),3),
                                                          MeanWeighted = round(sum(Weighted_CER)/sum(Length), 3),
                                                          Median = round(median(CER),3),
                                                          Min = round(min(CER),3),
                                                          Max = round(max(CER),3),
                                                          StdDev = round(sd(CER),3),
                                                          'Correctly predicted labels (%)' = round(mean(Correct),3)*100) %>%
  as.data.frame()

cer_plot <- ggplot(benchmark, aes(x = Lines, y = CER, fill = Lines)) +
  geom_violin(width = 1) +
  geom_boxplot(width = 0.05, color = "black", alpha = 0.2) +
  xlab("Number of lines") +
  ylab("CER") +
  ylim(c(0,1)) +
  labs(title = "Model performance by number of lines", 
       subtitle = paste0("CER over ", nrow(benchmark), " test examples"),
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


# Error analysis based on annotation quality
benchmark$Quality <- factor(benchmark$Quality, levels=c("Standard", "Irregular"))
benchmark$CER <- round(benchmark$CER, 4)

summary_df <- benchmark %>% group_by(Quality) %>% summarize(Examples = n(),
                                                            Mean = round(mean(CER),3),
                                                            MeanWeighted = round(sum(Weighted_CER)/sum(Length), 3),
                                                            Median = round(median(CER),3),
                                                            Min = round(min(CER),3),
                                                            Max = round(max(CER),3),
                                                            StdDev = round(sd(CER),3),
                                                            'Correctly predicted labels (%)' = round(mean(Correct),3)*100) %>%
  as.data.frame()


cer_plot <- ggplot(benchmark, aes(x = Quality, y = CER, fill = Quality)) +
  geom_violin(width = 1) +
  geom_boxplot(width = 0.05, color = "black", alpha = 0.2) +
  xlab("Quality") +
  ylab("CER") +
  ylim(c(0,1)) +
  labs(title = "Model performance by annotation quality", 
       subtitle = paste0("CER over ", nrow(benchmark), " test examples"),
       caption = "*Irregular: Contains errors and pencil annotations"
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


# Error analysis based on label length
benchmark$Category <- factor(benchmark$Category, levels=c("Correct", "Incorrect"))
benchmark$CER <- round(benchmark$CER, 4)

summary_df <- benchmark %>% group_by(Category) %>% summarize(Examples = n(),
                                                            Mean = round(mean(Length),3),
                                                            Median = round(median(Length),3),
                                                            Min = round(min(Length),3),
                                                            Max = round(max(Length),3),
                                                            StdDev = round(sd(Length),3),
                                                            'Correctly predicted labels (%)' = round(mean(Correct),3)*100) %>%
  as.data.frame()


cer_plot <- ggplot(benchmark, aes(x = Category, y = Length, fill = Category)) +
  geom_violin(width = 0.5) +
  geom_boxplot(width = 0.2, color = "black", alpha = 0.2) +
  xlab("Prediction class") +
  ylab("Label length") +
  labs(title = "Label length distribution by prediction class", 
       subtitle = paste0("Measured over ", nrow(benchmark), " test examples")
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



# Error analysis based on beam width
beam_analysis <- read_excel("/path/to/beam_analysis_swin_bert.xlsx")
beam_analysis$Width <- factor(beam_analysis$Width, levels=sort(unique(beam_analysis$Width)))
beam_analysis<-beam_analysis[beam_analysis$Split == "Test",]

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


