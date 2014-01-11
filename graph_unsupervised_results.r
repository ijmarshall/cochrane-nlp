
# graph unsupervised results


library(ggplot2)
library(grid)
library(gridExtra)


df <- read.csv('results/unsupervised.csv')

df$fold <- as.factor(df$fold)




# graph <- ggplot(df[df$fold==4,], aes(n)) + 
#   geom_line(aes(y = recall, colour = "recall")) + 
#   geom_line(aes(y = precision, colour = "precision"))


f1 <- ggplot(df[df$fold==4,], aes(n, f1)) + geom_line(color='steelblue', size=0.75) + xlab("iterations") + ylab(expression(f[1])) 



pdf("graph2.pdf", width=4, height=4)
print(f1)
dev.off()


# recall <- ggplot(df, aes(n, recall, color=fold)) + geom_line() 
# precision <- ggplot(df, aes(n, precision, color=fold)) + geom_line() 





# pdf("graph2.pdf", width=4, height=7)
# print(grid.arrange(f1, recall, precision))
# dev.off()




