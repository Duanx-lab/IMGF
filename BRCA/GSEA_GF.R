  library(limma)
  library(ROCR)
  
  #lab<-sc1
  #brmv_lab<-br_lab
  
  
  labs<-sf2
  #GSEtotal<-t(mrna)
  BRCA_exp<-BRCAGE
  
  GSEtotal<-t(BRCA_exp)
  labs[which(labs != 1)]<-0
  group<-labs
  group <- as.factor(group)
  
  fit <- lmFit(GSEtotal, model.matrix(~ group))
  fit <- eBayes(fit)
  tt <- topTable(fit, coef=2, number=Inf, adjust.method="BH", p.value = 0.01, lfc = 0.1)
  #tt <- tt[which(tt$P.Value < 0.05), ]
  de_genes <- rownames(tt)
  de_genes
  
  
  #au4<-de_genes
  #au3<-de_genes
  #au2<-de_genes
  #au1<-de_genes
  
  uni_genes <- de_genes
  target_pred  <- t(GSEtotal[which(rownames(GSEtotal) %in% uni_genes), ])
  # label <- as.vector(group)
  target_class <- matrix(rep(labs, length(uni_genes)), nrow = ncol(GSEtotal))
  pred <- prediction(target_pred, target_class)
  auc <- performance(pred,"auc")
  auc <- unlist(slot(auc, "y.values"))

  names(auc) <- colnames(target_pred)

  aucc<-auc[which(auc > 0.5)]

  #auc <- sort(auc[which(auc > 0.75)], decreasing = T)
  auc_genes <- names(aucc)
 
  auc_genes <- as.vector(auc_genes)
  
  auc_genes
  
  gene1<-auc_genes
  #gene2<-auc_genes
  #gene3<-auc_genes
 # gene4<-auc_genes
  #gene5<-auc_genes
  
  # 
  library(dplyr)
  library(ggplot2)
  library(gProfileR) 
  
  gsea <- gProfileR::gprofiler(gene1)
  
  gsea <- gsea %>% arrange(p.value)

  gsea <- gsea %>% arrange(p.value)
  df.plot <- gsea %>% mutate(GeneRatio=overlap.size/query.size) %>% 
    
    dplyr::select(Term=term.name, Count=overlap.size, GeneRatio, P=p.value) #%>% arrange((P))
  
  
  df.plot[1:20,]
  
  
  #dfm1<-df.plot
  #dfm2<-df.plot
  #dfm3<-df.plot
  
  #dfm4<-df.plot
  
  #dfm5<-df.plot
  #dfm6<-df.plot
  
  save(dfm1,dfm2,dfm3,dfm4,dfm5,file = "E:/DNNSD/pathway.RData")
  
  
  # dfv1<-dfm1[c(3,4,5,32,39,285),]
  # 
  # dfv2<-dfm2[c(1,5,43,46,26,120),]
  # 
  # dfv3<-dfm3[c(325,334,19,25,130,179),]
  # 
  # dfv4<-dfm4[c(604,581,16,442,17,1796),]
  # 
  # dfv5<-dfm5[c(129,179,176,226,317,403),]
  
  dfv1<-dff1[c(1,2,1413,2428,8,518),]
  
  dfv2<-dff2[c(128,70,45,34,1696,2258),]
  
  dfv3<-dff3[c(1,4,5,462,2087,2091),]
  
  dfv4<-dff4[c(7,8,10,11,427,30),]
  
  dfv5<-dff5[c(1:4,1054,27),]
  
 
   dfv1$C1<-rep("CS1",6)
   dfv2$C1<-rep("CS2",6)
   dfv3$C1<-rep("CS3",6)
   dfv4$C1<-rep("CS4",6)
   dfv5$C1<-rep("CS5",6)
    
   dr<-rbind(dfv1,dfv2,dfv3,dfv4,dfv5)
   
   dr$log10p<- -log10(dr$pvalue)
   
   library(reshape2)
   
   #mdc<-melt(dr,variable.name = "C1", value.name = "Count")
   
   col<-rev(c("#C4D3E1","#A6BCD2","#939BBF","#86609D","#731D6F"))
 
   
   library(showtext)
   font_add('Arial','/Library/Fonts/Arial.ttf') #???????壬MAC ?????????? /Library/Fonts
   showtext_auto() #?Զ?????showtext???????޷???ggsave()??ʹ?ã???Ϊggsave???Զ??򿪺͹ر?ͼ???豸??
   
   decimals <- sapply(dr$GeneRatio, function(x) eval(parse(text = x)))
   
   dr$GeneRatio<-round(decimals,3)
   
   pdf(file = "D:/gsea.pdf",width = 14,height = 9)
   
   ggplot() + geom_col(data =dr, aes(x = Description, y = GeneRatio,fill = log10p),width=0.7)+ facet_grid(. ~ C1,scales = 'free_x')+coord_cartesian(ylim=c(0,0.2))+
     scale_fill_gradient(limits=c(0.1, 12), low="slategray1",high = "slategray4")+geom_text(data=dr,aes(x= Description, y = GeneRatio , label =GeneRatio), position=position_dodge(width = 0.4),size = 3, vjust = -0.5,colour="blue")+
     theme( axis.text.x = element_text(size=18,angle = -60, hjust = 0),text=element_text(family = "Arial"),legend.title = element_text("-log10(p)"),
            axis.title.x=element_blank())
   
   dev.off() 
   
   #facet_wrap(~C1, scales = 'free_x')
   
   library(extrafont)
   
   library(showtext)
   font_add('Arial','/Library/Fonts/Arial.ttf') 
   showtext_auto() 
   
   
   pdf(file = "D:/dx/MOSD/figure/brca/gsea2.pdf",width = 10,height = 6)
   
   ggplot() + geom_point(data=dr,aes(x = Term, y = GeneRatio, size=Count, color=-log10(P))) + facet_wrap(~C1, scales = 'free_x')+theme_bw()+
     theme(axis.text.x = element_text(angle = -30, hjust = 0),text=element_text(family = "Arial"))+theme(text = element_text(size = 15)) +
    scale_color_gradient(low="darkturquoise",high="darkmagenta")
   
  
   dev.off()
   #df1<-df.plot
   
   #df2<-df.plot
   
   #df3<-df.plot
   
   #df4<-df.plot
   
   #df5<-df.plot
   
   #df6<-df.plot
   
   dfc1<-df1[c(1,2,4,18,25,29),]
   
   dfc2<-df2[c(30,35,38,61,110,153),]
   
   dfc3<-df3[c(1,2,3,6,9,39),]
   
   dfc4<-df4[c(1:6),]
   
   dfc5<-df5[c(3,6,7,11,13,14),]
   
   dfc6<-df6[c(3,5,6,8,15,18),]
   
   save(dfc1,dfc2,dfc3,dfc4,dfc5,dfc6,file = "E:/DNNSD/result/pathway.RData")
   
   
   
   
   library(clusterProfiler)
   library(org.Hs.eg.db)
   library(AnnotationDbi)
   
   # Symbol ?? Entrez ID
   gene_entrez <- bitr(gene5, 
                       fromType = "SYMBOL",
                       toType = c("ENTREZID", "ENSEMBL"),
                       OrgDb = org.Hs.eg.db)
   
   library(clusterProfiler)
   library(enrichplot)
   
   # ========== GO???? ==========
   ego <- enrichGO(gene         = gene_entrez$ENTREZID,
                   OrgDb        = org.Hs.eg.db,
                   keyType      = "ENTREZID",
                   ont          = "BP",       # BP/CC/MF
                   pAdjustMethod = "BH",
                   pvalueCutoff  = 0.05,
                   qvalueCutoff  = 0.2,
                   readable      = TRUE)      # ????gene symbol
   
  