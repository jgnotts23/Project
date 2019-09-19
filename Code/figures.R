#!/usr/bin/Rscript
# Author - Jacob Griffiths, jacob.griffiths18@imperial.ac.uk
# Date - Aug 2019

rm(list=ls())
graphics.off()

library(ggplot2)

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}


savePlot <- function(myPlot, plotname) {
  pdf(plotname)
  print(myPlot)
  dev.off()
}


timedata <- read.csv(file="/media/jacob/Samsung_external/time_data.csv", header=TRUE, sep=",")
timedata_confirmed <- read.csv(file="/media/jacob/Samsung_external/time_data_confirmed.csv", header=TRUE, sep=",")
    

# Weekday analysis
weekdays <- as.data.frame(table(timedata['Day_of_week']))
weekdays_confirmed <- as.data.frame(table(timedata_confirmed['Day_of_week']))

days <- c("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

counts <- weekdays[-1]
row.names(counts) <- weekdays$Var1

counts_confirmed <- weekdays_confirmed[-1]
row.names(counts_confirmed) <- weekdays_confirmed$Var1

frequencies <- (counts_confirmed / counts) * 100

frequencies <- frequencies[days,]

dat <- data.frame(
  day_of_week = factor(days, levels=days),
  gunshot_frequency = frequencies
)

theme_set(theme_gray(base_size = 30))
weekday_plot <- ggplot(data=dat, aes(x=day_of_week, y=gunshot_frequency, fill=day_of_week)) + 
  geom_bar(colour="black", fill="grey60", width=.6, stat="identity") + 
  guides(fill=FALSE) +
  xlab("Day of week") + ylab("Percentage of clips containing gunshot") + 
  theme(axis.text=element_text(size=12),  axis.title=element_text(size=12,face="bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
  panel.background = element_blank(), axis.line = element_line(colour = "black")) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

savePlot(weekday_plot, 'Figures/weekday_plot.pdf')


# Time of day analysis
hours <- as.data.frame(table(timedata['Hour']))
hours_confirmed <- as.data.frame(table(timedata_confirmed['Hour']))

hours_list <- c(0, 1, 2, 5, 6, 7, 8, 9, 14, 15, 16, 17, 18, 21, 22, 23)

counts <- hours[-1]
row.names(counts) <- hours$Var1

counts_confirmed <- hours_confirmed[-1]
row.names(counts_confirmed) <- hours_confirmed$Var1

frequencies <- (counts_confirmed / counts) * 100

#frequencies <- frequencies[hours_list,]

dat <- data.frame(
  hour_of_day = hours_list,
  gunshot_frequency = frequencies
  )


x <- ggplot(dat, aes(as.numeric(hour_of_day), Freq)) +
  geom_point() +
  #scale_x_continuous(breaks = seq(min(as.numeric(dat$hour_of_day)), max(as.numeric(dat$hour_of_day)), by = 1)) +
  theme_minimal()


## Statistical tests ##
#expected <- c(35.143, 35.143, 35.143, 35.143, 35.143, 35.143, 35.143)
audio_proportion <- weekdays[2] / 452605
expected_gunshots <- audio_proportion * 246

chi_data <- data.frame(
  "Day_of_week" = weekdays[1],
  "audio_proportion" = audio_proportion,
  "Total" = weekdays[2],
  "Observed" = weekdays_confirmed[2],
  "Expected" = expected_gunshots
)

colnames(chi_data) <- c("Day_of_week", "Audio_proportion", "Total", "Observed", "Expected")

result <- chisq.test(as.numeric(as.character(unlist(chi_data[[4]]))), as.numeric(as.character(unlist(chi_data[[2]]))))

## Time of day ##
hour_counts <- as.data.frame(table(timedata['Hour']))
night_counts <- hour_counts[c(1:3, 14:16),]
morning_counts <- hour_counts[4:8,]
afternoon_counts <- hour_counts[9:13,]

hour_counts_confirmed <- as.data.frame(table(timedata_confirmed['Hour']))
night_counts_confirmed <- hour_counts_confirmed[c(1:3, 14:16),]
morning_counts_confirmed <- hour_counts_confirmed[4:8,]
afternoon_counts_confirmed <- hour_counts_confirmed[9:13,]

total_night_clips <- sum(night_counts["Freq"])
total_morning_clips <- sum(morning_counts["Freq"])
total_afternoon_clips <- sum(afternoon_counts["Freq"])

total_night_confirmed <- sum(night_counts_confirmed["Freq"])
total_morning_confirmed <- sum(morning_counts_confirmed["Freq"])
total_afternoon_confirmed <- sum(afternoon_counts_confirmed["Freq"])

day_periods = c("Morning", "Afternoon", "Night")
total_clips = sum(total_night_clips) + sum(total_morning_clips) + sum(total_afternoon_clips)
day_period_counts = c(total_morning_clips, total_afternoon_clips, total_night_clips)
day_period_confirmed_counts = c(total_morning_confirmed, total_afternoon_confirmed, total_night_confirmed)


expected_gunshots <- audio_proportion * 246

chi_data <- data.frame(
  "Time_of_day" = day_periods,
  "audio_proportion" = audio_proportion,
  "Total" = day_period_counts,
  "Observed" = day_period_confirmed_counts,
  "Expected" = expected_gunshots
)

colnames(chi_data) <- c("Day_of_week", "Audio_proportion", "Total", "Observed", "Expected")

result <- chisq.test(as.numeric(as.character(unlist(chi_data[[4]]))), as.numeric(as.character(unlist(chi_data[[2]]))))


frequencies = (day_period_confirmed_counts / day_period_counts) * 100

dat <- data.frame(
  time_of_day = factor(day_periods, levels=day_periods),
  gunshot_frequency = frequencies
)

theme_set(theme_gray(base_size = 30))
timeday_plot <- ggplot(data=dat, aes(x=time_of_day, y=gunshot_frequency, fill=time_of_day)) + 
  geom_bar(colour="black", fill="grey60", width=.6, stat="identity") + 
  guides(fill=FALSE) +
  xlab("Time of day") + ylab("Percentage of clips containing gunshot") + 
  theme(axis.text=element_text(size=12),  axis.title=element_text(size=12,face="bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) 
  

savePlot(timeday_plot, 'Figures/timeday_plot.pdf')



## Area ##
area_counts <- as.data.frame(table(timedata['Area']))
area_counts_confirmed <- as.data.frame(table(timedata_confirmed['Area']))

areas = c("Indigenous_reserve", "La_Balsa", "Rancho_bajo", "SQ258", "SQ282", "SQ283")
total_clips = 452605

audio_proportion = area_counts["Freq"] / total_clips

expected_gunshots <- audio_proportion * 246

chi_data <- data.frame(
  "Area" = areas,
  "audio_proportion" = audio_proportion,
  "Total" = area_counts["Freq"],
  "Observed" = area_counts_confirmed["Freq"],
  "Expected" = expected_gunshots
)

colnames(chi_data) <- c("Area", "Audio_proportion", "Total", "Observed", "Expected")

result <- chisq.test(as.numeric(as.character(unlist(chi_data[[4]]))), as.numeric(as.character(unlist(chi_data[[2]]))))


frequencies = (chi_data["Observed"] / chi_data["Total"]) * 100

dat <- data.frame(
  Area = factor(areas, levels=areas),
  gunshot_frequency = frequencies
)

colnames(dat) <- c("Area", "gunshot_frequency")


area.color <- c("blue4", "red3", "blue4", "blue4", "blue4", "blue4")

theme_set(theme_gray(base_size = 30))

xticks <- c("Indigenous \n Reserve", "La Balsa", "Rancho \n bajo", "SQ258", "SQ282", "SQ283")

area_plot <- ggplot(data=dat, aes(x=Area, y=gunshot_frequency, fill=Area)) + 
  geom_bar(colour="black", fill=area.color, width=.6, stat="identity") + 
  guides(fill=FALSE) +
  xlab("Area") + ylab("Percentage of clips containing gunshot") + 
  scale_x_discrete(labels= xticks) +
  theme(axis.text=element_text(size=12),  axis.title=element_text(size=12,face="bold"), panel.grid.major = element_blank(), panel.grid.minor = element_blank(),
        panel.background = element_blank(), axis.line = element_line(colour = "black")) 

savePlot(area_plot, 'Figures/area_plot.pdf')

multiplot(weekday_plot, timeday_plot, area_plot, cols=2)



library(ggpubr)

ggarrange(weekday_plot, timeday_plot, area_plot + rremove("x.text"), 
          labels = c("A", "B", "C"),
          ncol = 2, nrow = 2)

weekday_plot
timeday_plot
area_plot
