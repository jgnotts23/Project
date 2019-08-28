#!/usr/bin/Rscript
# Author - Jacob Griffiths, jacob.griffiths18@imperial.ac.uk
# Date - Aug 2019

rm(list=ls())
graphics.off()

library(data.table)

library(dplyr)

library(formattable)

library(tidyr)

model_data <- read.csv(file="~/Documents/Project/Results/model_comparison.csv", row.names = 1, header=TRUE, sep=",")
model_data <- fread("~/Documents/Project/Results/model_comparison.csv", data.table=FALSE, header = TRUE, stringsAsFactors = FALSE)
colnames(model_data) = c("Area", "Belize", "Osa Peninsula", "Combined", "False Positives")
attach(model_data)

formattable(model_data)


#1)  First Data Table

formattable(model_data, 
            align =c("l","c","c","c","c", "c", "c", "c", "r"), 
            list(`V1` = formatter(
              "span", style = ~ style(color = "grey",font.weight = "bold")) 
            ))


