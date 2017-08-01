# check download file
baseDirProj = "/Users/weili/coursera/CapstoneProject/"
if (!file.exists(paste(baseDirProj,"Coursera-SwiftKey.zip", sep=""))) {
    download.file("https://d396qusza40orc.cloudfront.net/dsscapstone/dataset/Coursera-SwiftKey.zip")
    unzip("Coursera-SwiftKey.zip")
}

sourceDir = baseDirProj
listDirSource <- list.dirs(sourceDir)
idxEnUs <- grep('US',listDirSource )
listFileEnUS <- list.files(listDirSource[idxEnUs])

# get file's info
setwd(listDirSource[idxEnUs])
file.size = vector()
for (n in 1:length(listFileEnUS)){
    ext <- grepl("*.txt$", listFileEnUS[n])[1]
    if(ext){
        file.size[n] <- file.size(listFileEnUS[n])
    }
} 

for(ite in 1:3){
    print(ite)
    fileName=listFileEnUS[ite]
    con = file(fileName, "r")
    textThis = readLines(con)
    close(con)
    
    assign(paste0("nchars",ite) , lapply(textThis, nchar))
    assign(paste0("nwords",ite) , sapply(strsplit(textThis, "\\s+"), length))
    rm(textThis)
}
par(mar=c(1,1,1,1))

nchars1 <- as.numeric(unlist(nchars1))
nchars2 <- as.numeric(unlist(nchars2))
nchars3 <- as.numeric(unlist(nchars3))
nwords1 <- as.numeric(unlist(nwords1))
nwords2 <- as.numeric(unlist(nwords2))
nwords3 <- as.numeric(unlist(nwords3))

require(magrittr)
library(ggplot2)
library(reshape2)  
hist(log(nwords1))

DF <- rbind( data.frame(dataset=1, obs=log(nwords1)),
             data.frame(dataset=2, obs=log(nwords2)),
             data.frame(dataset=3, obs=log(nwords3)))
library(ggpubr)
bxp <- ggboxplot(DF, x = "dataset", y = "obs", ylab = "log(word of numbers per line)" , 
                 color = "dataset", palette = "jco")


DF$dataset <- as.factor(DF$dataset)
histp <- ggplot(DF, aes(x=obs, fill=dataset)) +
    geom_histogram(binwidth=1, colour="black", position="dodge") +
    xlab("log(word of numbers per line)")+
    scale_fill_manual(breaks=1:4, values=c("light blue","yellow","gray"))

ggarrange(bxp, histp, ncol = 2, nrow = 1)


names = character(3)
for(ite in 1:3){
    namethis=strsplit(listFileEnUS[ite], ".", fixed = TRUE)
    names[ite]=namethis[[1]][2]
}
# Summary of the data sets
mydata <- data.frame(source = names,
                     file.size.MB = file.size,
                     num.lines = c(length(nwords1), length(nwords2), length(nwords3)),
                     num.words = c(sum(nwords1), sum(nwords2), sum(nwords3)),
                     mean.num.words = c(mean(nwords1), mean(nwords2), mean(nwords3)),
                     median.num.words = c(median(nwords1), median(nwords2), median(nwords3)),
                     max.num.words = c(max(nwords1), max(nwords2), max(nwords3)),
                     sum.num.chars = c(sum(nchars1), sum(nchars2), sum(nchars3)),
                     max.num.chars = c(max(nchars1), max(nchars2), max(nchars3)))

rm(nchars1, nchars2, nchars3, nwords1, nwords2, nwords3)

#  part II
print("Part II")
library(plyr)

library(knitr)
require(NLP)
library(tm)
library(stringi)
library(RWeka)

library(ggplot2)
require(NLP)
library(slam)


for(ite in 1:3){
    print(ite)
    con = file(listFileEnUS[ite], "r")
    ####
    fsuffix <- strsplit( listFileEnUS[ite],"." , fixed = TRUE)
    # cname <- file.path(listDirSource[ite])
    # assign(paste0("corpus_",fsuffix[[1]][2]), Corpus(DirSource(cname)))
    
    assign(paste0("text_",fsuffix[[1]][2]), readLines(con, skipNul = TRUE) )
    
    close(con)
}
set.seed(1234)
text_mix_sample <-  c(sample(text_blogs, length(text_blogs) * 0.1),
                      sample(text_news, length(text_news) * 0.1),
                      sample(text_twitter, length(text_twitter) * 0.1))
rm(text_blogs, text_news, text_twitter)

# get corpus
corpus_mix_sample <- VCorpus(VectorSource(text_mix_sample))
# clean data
toSpace <- content_transformer(function (x , pattern ) gsub(pattern, " ", x))
corpus_mix_sample <- tm_map(corpus_mix_sample, toSpace, "/")
corpus_mix_sample <- tm_map(corpus_mix_sample, toSpace, "@")
corpus_mix_sample <- tm_map(corpus_mix_sample, toSpace, "\\|")
#Convert the text to lower case
corpus_mix_sample <- tm_map(corpus_mix_sample, content_transformer(tolower))
# Remove numbers
corpus_mix_sample <- tm_map(corpus_mix_sample, removeNumbers)
# Remove english common stopwords
corpus_mix_sample <- tm_map(corpus_mix_sample, removeWords, stopwords("english"))
# # Remove your own stop word
# # specify your stopwords as a character vector
# corpus_mix_sample <- tm_map(corpus_mix_sample, removeWords, c("blabla1", "blabla2"))
# Remove punctuations
corpus_mix_sample <- tm_map(corpus_mix_sample, removePunctuation)
# Eliminate extra white spaces
corpus_mix_sample <- tm_map(corpus_mix_sample, stripWhitespace)
# define get frequency function
getFreq <- function(tdm) {
    freq <- sort(rowSums(as.matrix(tdm)), decreasing = TRUE)
    return(data.frame(word = names(freq), freq = freq))
}
getFreq <- function(tdm) {
    freq <- sort(rowSums(as.matrix(rollup(tdm, 2, FUN = sum)), na.rm = T), decreasing = TRUE)
    return(data.frame(word = names(freq), freq = freq))
}
set.seed(1234)

bigram <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
trigram <- function(x) NGramTokenizer(x, Weka_control(min = 3, max = 3))

freq1 <- getFreq(removeSparseTerms(TermDocumentMatrix(corpus_mix_sample), 0.999))
freq2 <- getFreq(TermDocumentMatrix(corpus_mix_sample, control = list(tokenize = bigram, bounds = list(global = c(5, Inf)))))
freq3 <- getFreq(TermDocumentMatrix(corpus_mix_sample, control = list(tokenize = trigram, bounds = list(global = c(4, Inf)))))

# display
numTop=20
p1 <- ggplot(head(freq1,numTop), aes(x=reorder(word, -freq), y = freq))+
    geom_bar(stat="Identity", fill="light blue") +
    geom_text(aes(label=freq), vjust = -0.5) +
    ggtitle("Unigrams frequency") +
    ylab("Frequency") +
    xlab(paste("Top",numTop,"unigram sequence",sep=" "))+
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
p2 <- ggplot(head(freq2,numTop), aes(x=reorder(word, -freq), y = freq))+
    geom_bar(stat="Identity", fill="red") +
    geom_text(aes(label=freq), vjust = -0.5) +
    ggtitle("Bigram frequency sequence") +
    ylab("Frequency") +
    xlab(paste("Top",numTop,"bigram sequence",sep=" "))+
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
p3 <- ggplot(head(freq3,numTop), aes(x=reorder(word, -freq), y = freq))+
    geom_bar(stat="Identity", fill="light green") +
    geom_text(aes(label=freq), vjust = -0.5) +
    ggtitle("Trigram frequency  sequence") +
    ylab("Frequency") +
    xlab(paste("Top",numTop,"trigram sequence",sep=" ") )+
    theme(axis.text.x = element_text(angle = 90, hjust = 1))
p1
p2
p3

save(list=c("bxp", "histp","mydata", "p1", "p2", "p3","freq1", "freq2", "freq3" ), file="objectsOnly2v2.RData")


