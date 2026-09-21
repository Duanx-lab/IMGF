
library(limma)
library(HTSanalyzeR2)
library(dplyr)
library(cowplot)



## generate genesets
library(org.Hs.eg.db)
nf <- max(count.fields("F:/dx/Lssnf/breast/20201022/GSEA/genesets.gmt", sep = "\t"))
geneset <- read.table("F:/dx/Lssnf/breast/20201022/GSEA/genesets.gmt", 
                      stringsAsFactors = F, fill = T, col.names = 1:nf)
rslt <- lapply(1:nrow(geneset), function(i) {
  gs1 <- geneset[i,, drop = T] %>% unlist
  gs1 <- gs1[2:length(gs1)] 
  gs1 <- gs1[gs1 != ""]
  gs1 <- mapIds(org.Hs.eg.db, keys = gs1, keytype = "SYMBOL", column = "ENTREZID")
  names(gs1) <- NULL
  gs1 <- gs1[!is.na(gs1)]
  gs1
})
names(rslt) <- geneset$X1
geneset <- rslt


#load("D:/dx/RData/data.RData")


#load("D:/dx/RData/ws.RData")
# symbols <- mapIds(org.Hs.eg.db, keys = ensemblsIDS, keytype = "ENSEMBL", column="SYMBOL")
# 
# mapr<-mapIds(org.Hs.eg.db, keys = colnames(ga253), keytype = "SYMBOL", column = "ENTREZID")
# sym<-sapply(1:length(colnames(mydatGE)), function(i){
#   strsplit(colnames(mydatGE)[i],"|",fixed = TRUE)[[1]][1]
# })
# 
# mydatGE1<-mydatGE[,-c(1:30)]
# 
# sym1<-sym[-c(1:30)]
# 
# colnames(mydatGE1)<-sym1

load("D:/project/MOSD/data/BRCA_used_omics.RData")

data<-t(BRCAGE)
ws<-lab

#data<-data[c(1:16000,17000:20501),]

names(ws)<-colnames(data)

identical(names(ws), colnames(data))
## class1 VS others
group <- rep("0", ncol(data))
group[which(ws == "1")] <- "1"
group <- as.factor(group)

fit <- lmFit(data, model.matrix(~ group))
fit <- eBayes(fit)
class1.limma <- topTable(fit, coef=2, number=Inf, adjust.method="BH")

## class2 VS others
group <- rep("0", ncol(data))
group[which(ws == "2")] <- "1"
group <- as.factor(group)

fit <- lmFit(data, model.matrix(~ group))
fit <- eBayes(fit)
class2.limma <- topTable(fit, coef=2, number=Inf, adjust.method="BH")

## class3 VS others
group <- rep("0", ncol(data))
group[which(ws == "3")] <- "1"
group <- as.factor(group)

fit <- lmFit(data, model.matrix(~ group))
fit <- eBayes(fit)
class3.limma <- topTable(fit, coef=2, number=Inf, adjust.method="BH")

#class4 VS others
group <- rep("0", ncol(data))

group[which(ws == "4")] <- "1"
group <- as.factor(group)

fit <- lmFit(data, model.matrix(~ group))
fit <- eBayes(fit)
class4.limma <- topTable(fit, coef=2, number=Inf, adjust.method="BH")

#class5 VS others
group <- rep("0", ncol(data))

group[which(ws == "5")] <- "1"
group <- as.factor(group)

fit <- lmFit(data, model.matrix(~ group))
fit <- eBayes(fit)
class5.limma <- topTable(fit, coef=2, number=Inf, adjust.method="BH")

#class6 VS others
group <- rep("0", ncol(data))

group[which(ws == "6")] <- "1"
group <- as.factor(group)

fit <- lmFit(data, model.matrix(~ group))
fit <- eBayes(fit)
class6.limma <- topTable(fit, coef=2, number=Inf, adjust.method="BH")

# 
# limma.rslt <- list(class1.limma, class2.limma, class3.limma, class4.limma)
# names(limma.rslt) <- c("class1.limma", "class2.limma", "class3.limma", "class4.limma")

limma.rslt <- list(class1.limma, class2.limma, class3.limma,class4.limma, class5.limma, class6.limma)
names(limma.rslt) <- c("class1.limma", "class2.limma", "class3.limma","class4.limma", "class5.limma", "class6.limma")


## GSEA

ListGSC <- list(geneset = geneset)
GSCA.rslt <- lapply(limma.rslt, function(x) {
  data4enrich <- x$logFC
  names(data4enrich) <- rownames(x)
  gsca <- GSCA(listOfGeneSetCollections = ListGSC, geneList = data4enrich)
  gsca1 <- preprocess(gsca, species="Hs", initialIDs="SYMBOL", keepMultipleMappings=TRUE,
                      duplicateRemoverMethod="max", orderAbsValue=FALSE)
  doParallel::registerDoParallel(cores=2)
  gsca2 <- analyze(gsca1, para=list(pValueCutoff=0.05, pAdjustMethod ="BH",
                                    nPermutations=100000, minGeneSetSize=1, exponent=1),
                   doGSOA = F, doGSEA = TRUE)
  gsca3 <- appendGSTerms(gsca2, msigdbGSCs='geneset')
  gsca3
})
names(GSCA.rslt) <- names(limma.rslt)


save(GSCA.rslt, file="D:/project/OMSP/result/gsea/data.gsca.RData")

# Allgeneset <- GSCA.rslt[[1]]@listOfGeneSetCollections$geneset %>% names
# specificGeneset <- list("geneset"=Allgeneset)
# reportAll(GSCA.rslt, specificGeneset = specificGeneset)

GSEA.rslt <- lapply(GSCA.rslt, function(x){
  x@result$GSEA.results$geneset
})
save(GSEA.rslt, file="D:/project/OMSP/result/gsea/data.gsea.RData")

## heatmap

rm(list = ls())

load("D:/project/OMSP/result/gsea/data.gsea.RData")

class1.gsea <- GSEA.rslt[[1]] %>% as.data.frame()
class1.gsea$Pvalue <- -log10(class1.gsea$Pvalue + 10^-5)
for(i in 1:nrow(class1.gsea)){
  if(class1.gsea[i, "Observed.score"] < 0){
    class1.gsea[i, "Pvalue"] <- -class1.gsea[i, "Pvalue"]
  }
}

class2.gsea <- GSEA.rslt[[2]] %>% as.data.frame()
class2.gsea$Pvalue <- -log10(class2.gsea$Pvalue + 10^-5)
for(i in 1:nrow(class2.gsea)){
  if(class2.gsea[i, "Observed.score"] < 0){
    class2.gsea[i, "Pvalue"] <- -class2.gsea[i, "Pvalue"]
  }
}
class2.gsea <- class2.gsea[match(rownames(class1.gsea), rownames(class2.gsea)), ]

class3.gsea <- GSEA.rslt[[3]] %>% as.data.frame()
class3.gsea$Pvalue <- -log10(class3.gsea$Pvalue + 10^-5)
for(i in 1:nrow(class3.gsea)){
  if(class3.gsea[i, "Observed.score"] < 0){
    class3.gsea[i, "Pvalue"] <- -class3.gsea[i, "Pvalue"]
  }
}
class3.gsea <- class3.gsea[match(rownames(class1.gsea), rownames(class3.gsea)), ]

class4.gsea <- GSEA.rslt[[4]] %>% as.data.frame()
class4.gsea$Pvalue <- -log10(class4.gsea$Pvalue + 10^-5)
for(i in 1:nrow(class4.gsea)){
  if(class4.gsea[i, "Observed.score"] < 0){
    class4.gsea[i, "Pvalue"] <- -class4.gsea[i, "Pvalue"]
  }
}
class4.gsea <- class4.gsea[match(rownames(class1.gsea), rownames(class4.gsea)), ]

class5.gsea <- GSEA.rslt[[5]] %>% as.data.frame()
class5.gsea$Pvalue <- -log10(class5.gsea$Pvalue + 10^-5)
for(i in 1:nrow(class5.gsea)){
  if(class5.gsea[i, "Observed.score"] < 0){
    class5.gsea[i, "Pvalue"] <- -class5.gsea[i, "Pvalue"]
  }
}
class5.gsea <- class5.gsea[match(rownames(class1.gsea), rownames(class5.gsea)), ]


class6.gsea <- GSEA.rslt[[6]] %>% as.data.frame()
class6.gsea$Pvalue <- -log10(class6.gsea$Pvalue + 10^-5)
for(i in 1:nrow(class6.gsea)){
  if(class6.gsea[i, "Observed.score"] < 0){
    class6.gsea[i, "Pvalue"] <- -class6.gsea[i, "Pvalue"]
  }
}
class6.gsea <- class6.gsea[match(rownames(class1.gsea), rownames(class6.gsea)), ]



## merge
# heat.dat <- cbind(class1.gsea$Pvalue, class2.gsea$Pvalue, 
#                   class3.gsea$Pvalue, class6.gsea$Pvalue)

heat.dat <- cbind(class1.gsea$Pvalue, class2.gsea$Pvalue, 
                  class3.gsea$Pvalue,class4.gsea$Pvalue,class5.gsea$Pvalue,class6.gsea$Pvalue)
rownames(heat.dat) <- rownames(class1.gsea)
colnames(heat.dat) <- paste0("Class", 1:6)
save(heat.dat, file = "D:/project/OMSP/result/gsea/pathway.heat.RData")

## rename pathways
## Signature.heat
load("D:/project/OMSP/result/gsea/pathway.heat.RData")



heat.dat[heat.dat <= -2] <- -2
heat.dat[heat.dat >= 2] <- 2
Signature.heat <- heat.dat[match(c("EPITH_LOBODA",
                                   "WNT_FLIER",
                                   "MYC_TARGETS_ZELLER",
                                   "MESENCH_LOBODA",
                                   "EMT_CORE_GENES",
                                   "TGFB_KEGG",
                                   "MATRIX_REMODEL_REACTOME",
                                   "WOUND_RESPONSE_GO_BP",
                                   "CSC_BATLLE"), rownames(heat.dat)), ]

rownames(Signature.heat) <- c("Epithelial", "WNT targets", "MYC targets", "Mesenchymal",
                              "EMT activation", "TGFB-activation",
                              "Matrix remodeling", "Wound response", "Cancer stem cell")
## Pathways.heat
Pathways.heat <- heat.dat[match(c("MAPK_KEGG",
                                  "PI3K_ACT_REACTOME",
                                  "SRC_ACT_BILD",
                                  "JAK_STAT_KEGG",
                                  "CASPASE_BIOCARTA",
                                  "PROTEASOME_KEGG",
                                  "KEGG_CELL_CYCLE",
                                  "TRANSLATION_RIBOS_REACTOME",
                                  "INTEGRIN_BETA3_CP",
                                  "VEGF_VEGFR_REACTOME"), rownames(heat.dat)), ]
rownames(Pathways.heat) <- c("MAPK", "PI3K", "SRC",
                             "JAK-STAT", "Caspases", "Proteosome", "Cell cycle",
                             "Translation ribosome", "Integrin-B3", "VEGF VEGFR")

## Estimate.heat
Estimate.heat <- heat.dat[match(c("IMMUNE_ESTIMATE",
                                  "STROMAL_ESTIMATE"), rownames(heat.dat)), ]
rownames(Estimate.heat) <- c("Immune infiltration", "Stromal infiltration")

## Immune.heat
Immune.heat <- heat.dat[match(c("IMMUNE_RESP_GO_BP",
                                "PD1_REACTOME",
                                "IMMUNE_NKC_BREAST",
                                "IMMUNE_TH1_GALON",
                                "IMMUNE_THF_BREAST",
                                "IMMUNE_TH17_GOUNARI",
                                "IMMUNE_TREG_GALON",
                                "COMPLEMENT_COAG_KEGG"), rownames(heat.dat)), ]

rownames(Immune.heat) <- c("Immune response","PD1 activation", "NK cell infiltration",
                           "TH1 infiltration","TFH infiltration", "TH17 activation",
                           "Treg activation", "Complement activation")

## Metabolism.heat
Metabolism.heat <- heat.dat[match(c("AMINO_SUGAR_NUCLEO_METAB_KEGG",
                                    "PENTOSE_GLUC_METAB_KEGG",
                                    "FRUTOSE_MANNOSE_METAB_KEGG",
                                    "GALACTOSE_METAB_KEGG",
                                    "GLUTAMINE_GO_BP",
                                    "GLUTATHIONE_KEGG",
                                    "NITROGEN_METAB_KEGG",
                                    "GLYCEROPHOSPHOLIPID_GO_BP",
                                    "LYSOPHOSPHOLIPID_PID",
                                    "FATTY_ACID_METAB_KEGG"), rownames(heat.dat)), ]
rownames(Metabolism.heat) <- c("Sugar aa nucleotide", "Glucose pentose",
                               "Fructose mannose", "Galactose", "Glutamine", "Glutathione",
                               "Nitrogen", "Glycerophospholipid", "Lysophospholipid", "Fatty acid")

## pheatmap
library(pheatmap)
col.annotation <- data.frame("group" = colnames(heat.dat),
                             "Class" = paste0("Class", 1:6))
annotation_col <- data.frame("Class" = as.factor(col.annotation$Class))
rownames(annotation_col) <- colnames(Metabolism.heat)
# ann_colors = list(Class = c(Class1= "#3B4992FF",Class2="#EE0000FF",
#                             Class3= "#008B45FF", Class4="#631879FF"))
#c("#E18727FF","#0072B5FF","#20854EFF","#BC3C29FF")
#c("#0073C2FF","#EFC000FF","#868686FF","#CD534CFF","#7AA6DCFF","#003C67FF","#8F7700FF","#3B3B3BFF")
ann_colors = list(Class = c(Class1= "#0073C2FF",Class2="#EFC000FF",
                            Class3= "#868686FF",Class4= "#CD534CFF",Class5= "#7AA6DCFF",Class6= "#003C67FF"
                           ))
#library(RColorBrewer)
#pal_jco("default")(4)
#brewer.pal(5,"RdBu")
#"steelblue", "#FFFFFF", "darkorange"
## Signature.heatmap
breaks <- c(seq(-2, 2, length.out = 100))
Signature.p <- pheatmap(Signature.heat, 
                        filename = "D:/project/OMSP/result/gsea/Signature.pdf",
                        fontsize =10, breaks = breaks,
                        show_rownames=T,annotation_col = annotation_col, annotation_colors = ann_colors,
                        color = colorRampPalette(c("#92C5DE", "#FFFFFF", "#F4A582"))(100),
                        # border_color = F,
                        annotation_names_col=F,show_colnames=F,
                        cluster_cols =F,cluster_rows=F,fontsize_col=12,
                        cellwidth = 20,cellheight=20,
                        main = "Signatures") 

## Pathways.heatmap
breaks <- c(seq(-2, 2, length.out = 100))
Pathways.p <- pheatmap(Pathways.heat, 
                       filename = "D:/project/OMSP/result/gsea/Pathways.pdf",
                       fontsize =10, breaks = breaks,
                       show_rownames=T,annotation_col = annotation_col, annotation_colors = ann_colors,
                       color = colorRampPalette(c("#92C5DE", "#FFFFFF", "#F4A582"))(100),
                       # border_color = F,
                       annotation_names_col=F,show_colnames=F,
                       cluster_cols =F,cluster_rows=F,fontsize_col=12,
                       cellwidth = 20,cellheight=20,
                       legend = F, annotation_legend = F,
                       main = "Pathways")

## Estimate.heatmap
Estimate.p <- pheatmap(Estimate.heat,
                       filename = "D:/project/OMSP/result/gsea/Estimate.pdf",
                       fontsize =10, breaks = breaks,
                       show_rownames=T,annotation_col = annotation_col, annotation_colors = ann_colors,
                       color = colorRampPalette(c("#92C5DE", "#FFFFFF", "#F4A582"))(100),
                       # border_color = F,
                       annotation_names_col=F,show_colnames=F,
                       cluster_cols =F,cluster_rows=F,fontsize_col=12,
                       cellwidth = 20,cellheight=20,
                       legend = F, annotation_legend = F, 
                       main = "Estimate")

## Immune.heatmap
Immune.p <- pheatmap(Immune.heat,
                     filename = "D:/project/OMSP/result/gsea/Immune.pdf",
                     fontsize =10, breaks = breaks,
                     show_rownames=T,annotation_col = annotation_col, annotation_colors = ann_colors,
                     color = colorRampPalette(c("#92C5DE", "#FFFFFF", "#F4A582"))(100),
                     # border_color = F,
                     annotation_names_col=F,show_colnames=F,
                     cluster_cols =F,cluster_rows=F,fontsize_col=12,
                     cellwidth = 20,cellheight=20,
                     legend = F, annotation_legend = F,
                     main = "Immune")

## Metabolism.heatmap
# breaks <- c(seq(-5, 5, length.out = 100))
Metabolism.p <- pheatmap(Metabolism.heat, 
                         filename = "D:/project/OMSP/result/gsea/Metabolism.pdf",
                         fontsize =10, breaks = breaks,
                         show_rownames=T,annotation_col = annotation_col, annotation_colors = ann_colors,
                         color = colorRampPalette(c("#92C5DE", "#FFFFFF", "#F4A582"))(100),
                         # border_color = F,
                         annotation_names_col=F,show_colnames=F,
                         cluster_cols =F,cluster_rows=F,fontsize_col=12,
                         cellwidth = 20,cellheight=20,
                         legend = F, annotation_legend = F,
                         main = "Metabolism")

save(Signature.p, Pathways.p, Immune.p, Metabolism.p, Estimate.p,
     file = "D:/project/OMSP/result/gsea/Figure.RData")
pdf("D:/project/OMSP/result/gsea/pathway.pdf", height = 12, width = 10)
plot_grid(Signature.p$gtable, Pathways.p$gtable, Estimate.p$gtable,
          Metabolism.p$gtable, Immune.p$gtable, labels = "auto",
          ncol = 2, rel_heights = c(2.5, 3, 0.8, 3, 2),
          align = "v")
dev.off()




