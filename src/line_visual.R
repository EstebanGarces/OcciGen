library(readr)
library(readxl)
#synthetic_train <- read_csv("C:/Users/ru84fuj/Desktop/results/3/synthetic_train.csv")
#benchmark <- read_excel("C:/Users/ru84fuj/Desktop/results/benchmark_own_vision_tesseract_40000.xlsx")
benchmark <- read_excel("C:/Users/ru84fuj/Desktop/help_files/num_lines_test.xlsx")


benchmark$Lines <- factor(benchmark$Lines, levels=unique(benchmark$Lines))
benchmark$CER <- round(benchmark$CER, 4)

# Focus on test set only
benchmark<-benchmark[benchmark$Split == "Test",]


library(ggplot2)
library(dplyr)
library(scales)
library(grid)
library(gridExtra)
library(ggthemes)

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
       #caption = "*Real data include inaccurate labels (10%) - relabeling ongoing"
  ) +
  theme_economist() +
  guides(fill = "none") +
  scale_fill_economist() +
  theme(
    axis.title = element_text(face = "bold", size = 14)
  )


caption = "*All predictions were performed over real images"


tt <- ttheme_default(colhead=list(fg_params = list(parse=TRUE)),
                     base_size = 12,
                     padding = unit(c(20, 4), "mm"))

tbl <- tableGrob(summary_df, rows=NULL, theme=tt)

grid.arrange(cer_plot, tbl, 
             nrow = 2, heights = c(4, 2))

grid.arrange(density_plot, tbl, 
             nrow = 2, heights = c(4, 2))


# Analysis by quality of image

library(readr)
library(readxl)
#synthetic_train <- read_csv("C:/Users/ru84fuj/Desktop/results/3/synthetic_train.csv")
#benchmark <- read_excel("C:/Users/ru84fuj/Desktop/results/benchmark_own_vision_tesseract_40000.xlsx")
benchmark <- read_excel("C:/Users/ru84fuj/Desktop/help_files/num_lines_test.xlsx")


benchmark$Quality <- factor(benchmark$Quality, levels=c("Standard", "Irregular"))
benchmark$CER <- round(benchmark$CER, 4)

# Focus on test set only
benchmark<-benchmark[benchmark$Split == "Test",]


library(ggplot2)
library(dplyr)
library(scales)
library(grid)
library(gridExtra)
library(ggthemes)

summary_df <- benchmark %>% group_by(Quality) %>% summarize(Examples = n(),
                                                            Mean = round(mean(CER),3),
                                                            MeanWeighted = round(sum(Weighted_CER)/sum(Length), 3),
                                                            Median = round(median(CER),3),
                                                            Min = round(min(CER),3),
                                                            Max = round(max(CER),3),
                                                            StdDev = round(sd(CER),3),
                                                            'Correctly predicted labels (%)' = round(mean(Correct),3)*100) %>%
  as.data.frame()


#colnames(summary_df)<-c("Dataset", "Number of examples", "Mean", "Median", "Min", "Max", "Standard Deviation")

# cer_plot<-ggplot(benchmark, aes(x = Quality, y = CER, fill = Quality))+geom_violin()+
#   xlab("Quality")+ylab("CER")+labs(title = "Model performance by annotation quality", 
#                                  subtitle = paste0("CER over ", nrow(benchmark), " test examples*"),
#                                  caption = "*Irregular: Contains errors and pencil annotations"
#   )+theme_economist() + scale_fill_economist()


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




# density_plot<-ggplot(benchmark, aes(x = CER, fill = Quality))+geom_density(alpha = 0.50)+
#   xlab("CER")+ylab("Frequency")+xlim(c(0,1))+labs(title = "Model performance by annotation quality", 
#                                                   subtitle = paste0("CER over ", nrow(benchmark), " test examples*"))+theme_economist() + scale_fill_economist()



caption = "*All predictions were performed over real images"



tt <- ttheme_default(colhead=list(fg_params = list(parse=TRUE)),
                     base_size = 12,
                     padding = unit(c(20, 4), "mm"))

tbl <- tableGrob(summary_df, rows=NULL, theme=tt)

grid.arrange(cer_plot, tbl, 
             nrow = 2, heights = c(4, 2))

grid.arrange(density_plot, tbl, 
             nrow = 2, heights = c(4, 2))



# Analysis by length


library(readr)
library(readxl)
#synthetic_train <- read_csv("C:/Users/ru84fuj/Desktop/results/3/synthetic_train.csv")
#benchmark <- read_excel("C:/Users/ru84fuj/Desktop/results/benchmark_own_vision_tesseract_40000.xlsx")
benchmark <- read_excel("C:/Users/ru84fuj/Desktop/help_files/num_lines_test.xlsx")


benchmark$Category <- factor(benchmark$Category, levels=c("Correct", "Incorrect"))
benchmark$CER <- round(benchmark$CER, 4)

# Focus on test set only
benchmark<-benchmark[benchmark$Split == "Test",]

# Exclude GPT2
#benchmark <-benchmark[benchmark$Model != 'Own (ViT+GPT2)' & benchmark$Model != 'Own (ViT+pretrained GPT2)',]

library(ggplot2)
library(dplyr)
library(scales)
library(grid)
library(gridExtra)
library(ggthemes)

summary_df <- benchmark %>% group_by(Category) %>% summarize(Examples = n(),
                                                            Mean = round(mean(Length),3),
                                                            #MeanWeighted = round(sum(Weighted_CER)/sum(nchar(Label)), 3),
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
  #ylim(c(0,1)) +
  labs(title = "Label length distribution by prediction class", 
       subtitle = paste0("Measured over ", nrow(benchmark), " test examples")
  ) +
  theme_economist() +
  guides(fill = "none") +
  scale_fill_economist() +
  theme(
    axis.title = element_text(face = "bold", size = 14)
  )

cer_plot

caption = "*All predictions were performed over real images"



tt <- ttheme_default(colhead=list(fg_params = list(parse=TRUE)),
                     base_size = 12,
                     padding = unit(c(20, 4), "mm"))

tbl <- tableGrob(summary_df, rows=NULL, theme=tt)

grid.arrange(cer_plot, tbl, 
             nrow = 2, heights = c(4, 2))

grid.arrange(density_plot, tbl, 
             nrow = 2, heights = c(4, 2))





