library(pROC)
library(tidyverse)



mergePath <- '/PATH_TO/data'
preds_1y_fname <- 'IHD_8139_preds_all_1y.csv'
preds_5y_fname <- 'IHD_8139_preds_all_5y.csv'

predictions_1y <- read_csv(paste(mergePath, preds_1y_fname, sep="/")) %>%
  filter(set=="test")
predictions_5y <- read_csv(paste(mergePath, preds_5y_fname, sep="/")) %>%
  filter(set=="test")


library(ggplot2)# Create multiple curves to plot
rocs_1y <- roc(label ~ frs + pce_risk + seg_risk + pce_seg_model_pred + img_pred + 
               clin_pred + img_clin_fusion_preds + img_clin_seg_fusion_preds, 
            data = predictions_1y,
            ci=TRUE)
rocs_5y <- roc(label ~ frs + pce_risk + seg_risk + pce_seg_model_pred + img_pred + 
                 clin_pred + img_clin_fusion_preds + img_clin_seg_fusion_preds,
               data = predictions_5y,
               ci=TRUE)

# AUC comparison with DeLong's method:
roc.test(rocs_1y$frs, rocs_1y$pce_risk)$p.value
roc.test(rocs_1y$frs, rocs_1y$seg_risk)$p.value
roc.test(rocs_1y$frs, rocs_1y$pce_seg_model_pred)$p.value
roc.test(rocs_1y$frs, rocs_1y$clin_pred)$p.value
roc.test(rocs_1y$frs, rocs_1y$img_pred)$p.value
roc.test(rocs_1y$frs, rocs_1y$img_clin_fusion_preds)$p.value
roc.test(rocs_1y$frs, rocs_1y$img_clin_seg_fusion_preds)$p.value

roc.test(rocs_1y$pce_risk, rocs_1y$seg_risk)$p.value
roc.test(rocs_1y$pce_risk, rocs_1y$pce_seg_model_pred)$p.value
roc.test(rocs_1y$pce_risk, rocs_1y$clin_pred)$p.value
roc.test(rocs_1y$pce_risk, rocs_1y$img_pred)$p.value
roc.test(rocs_1y$pce_risk, rocs_1y$img_clin_fusion_preds)$p.value
roc.test(rocs_1y$pce_risk, rocs_1y$img_clin_seg_fusion_preds)$p.value

roc.test(rocs_1y$seg_risk, rocs_1y$pce_seg_model_pred)$p.value
roc.test(rocs_1y$seg_risk, rocs_1y$clin_pred)$p.value
roc.test(rocs_1y$seg_risk, rocs_1y$img_pred)$p.value
roc.test(rocs_1y$seg_risk, rocs_1y$img_clin_fusion_preds)$p.value
roc.test(rocs_1y$seg_risk, rocs_1y$img_clin_seg_fusion_preds)$p.value

roc.test(rocs_1y$pce_seg_model_pred, rocs_1y$clin_pred)$p.value
roc.test(rocs_1y$pce_seg_model_pred, rocs_1y$img_pred)$p.value
roc.test(rocs_1y$pce_seg_model_pred, rocs_1y$img_clin_fusion_preds)$p.value
roc.test(rocs_1y$pce_seg_model_pred, rocs_1y$img_clin_seg_fusion_preds)$p.value

roc.test(rocs_1y$clin_pred, rocs_1y$img_pred)$p.value
roc.test(rocs_1y$clin_pred, rocs_1y$img_clin_fusion_preds)$p.value
roc.test(rocs_1y$clin_pred, rocs_1y$img_clin_seg_fusion_preds)$p.value

roc.test(rocs_1y$img_pred, rocs_1y$img_clin_fusion_preds)$p.value
roc.test(rocs_1y$img_pred, rocs_1y$img_clin_seg_fusion_preds)$p.value

roc.test(rocs_1y$img_clin_fusion_preds, rocs_1y$img_clin_seg_fusion_preds)$p.value

# 5y
roc.test(rocs_5y$frs, rocs_5y$pce_risk)$p.value
roc.test(rocs_5y$frs, rocs_5y$seg_risk)$p.value
roc.test(rocs_5y$frs, rocs_5y$pce_seg_model_pred)$p.value
roc.test(rocs_5y$frs, rocs_5y$clin_pred)$p.value
roc.test(rocs_5y$frs, rocs_5y$img_pred)$p.value
roc.test(rocs_5y$frs, rocs_5y$img_clin_fusion_preds)$p.value
roc.test(rocs_5y$frs, rocs_5y$img_clin_seg_fusion_preds)$p.value


roc.test(rocs_5y$pce_risk, rocs_5y$seg_risk)$p.value
roc.test(rocs_5y$pce_risk, rocs_5y$pce_seg_model_pred)$p.value
roc.test(rocs_5y$pce_risk, rocs_5y$clin_pred)$p.value
roc.test(rocs_5y$pce_risk, rocs_5y$img_pred)$p.value
roc.test(rocs_5y$pce_risk, rocs_5y$img_clin_fusion_preds)$p.value
roc.test(rocs_5y$pce_risk, rocs_5y$img_clin_seg_fusion_preds)$p.value

roc.test(rocs_5y$seg_risk, rocs_5y$pce_seg_model_pred)$p.value
roc.test(rocs_5y$seg_risk, rocs_5y$clin_pred)$p.value
roc.test(rocs_5y$seg_risk, rocs_5y$img_pred)$p.value
roc.test(rocs_5y$seg_risk, rocs_5y$img_clin_fusion_preds)$p.value
roc.test(rocs_5y$seg_risk, rocs_5y$img_clin_seg_fusion_preds)$p.value

roc.test(rocs_5y$pce_seg_model_pred, rocs_5y$clin_pred)$p.value
roc.test(rocs_5y$pce_seg_model_pred, rocs_5y$img_pred)$p.value
roc.test(rocs_5y$pce_seg_model_pred, rocs_5y$img_clin_fusion_preds)$p.value
roc.test(rocs_5y$pce_seg_model_pred, rocs_5y$img_clin_seg_fusion_preds)$p.value

roc.test(rocs_5y$clin_pred, rocs_5y$img_pred)$p.value
roc.test(rocs_5y$clin_pred, rocs_5y$img_clin_fusion_preds)$p.value
roc.test(rocs_5y$clin_pred, rocs_5y$img_clin_seg_fusion_preds)$p.value

roc.test(rocs_5y$img_pred, rocs_5y$img_clin_fusion_preds)$p.value
roc.test(rocs_5y$img_pred, rocs_5y$img_clin_seg_fusion_preds)$p.value

roc.test(rocs_5y$img_clin_fusion_preds, rocs_5y$img_clin_seg_fusion_preds)$p.value

ci_specs <- seq(0,1,l=50)
ci_objs_1y_fig <- list(
  ci.se(rocs_1y$pce_risk, specificities = ci_specs),
  ci.se(rocs_1y$img_clin_fusion_preds, specificities = ci_specs)
)
ci_objs_1y_fig2 <- list(
  ci.se(rocs_1y$frs, specificities = ci_specs),
  ci.se(rocs_1y$pce_risk, specificities = ci_specs),
  ci.se(rocs_1y$seg_risk, specificities = ci_specs),
  ci.se(rocs_1y$pce_seg_model_pred, specificities = ci_specs),
  ci.se(rocs_1y$img_pred, specificities = ci_specs),
  ci.se(rocs_1y$clin_pred, specificities = ci_specs),
  ci.se(rocs_1y$img_clin_fusion_preds, specificities = ci_specs),
  ci.se(rocs_1y$img_clin_seg_fusion_preds, specificities = ci_specs)
)

ci_objs_5y_fig <- list(
  ci.se(rocs_5y$pce_risk, specificities = ci_specs),
  ci.se(rocs_5y$img_clin_fusion_preds, specificities = ci_specs)
)
ci_objs_5y_fig2 <- list(
  ci.se(rocs_5y$frs, specificities = ci_specs),
  ci.se(rocs_5y$pce_risk, specificities = ci_specs),
  ci.se(rocs_5y$seg_risk, specificities = ci_specs),
  ci.se(rocs_5y$pce_seg_model_pred, specificities = ci_specs),
  ci.se(rocs_5y$img_pred, specificities = ci_specs),
  ci.se(rocs_5y$clin_pred, specificities = ci_specs),
  ci.se(rocs_5y$img_clin_fusion_preds, specificities = ci_specs),
  ci.se(rocs_5y$img_clin_seg_fusion_preds, specificities = ci_specs)
)

plot_multiple_roc_curve_with_ci <- function(roc_objs, ci_objs, names, line_colors, n_legend_cols=2) {
  
  for (i in 1:length(roc_objs)){
    sens <- roc_objs[[i]]$sensitivities
    spec <- roc_objs[[i]]$specificities
    
    ci_obj <- ci_objs[[i]]
    
    line_data_group <- tibble(source=names[i], sensitivity=sens, specificity=spec)
    ci_data_group <- tibble(source=names[i], spec = ci_specs, low=unname(ci_obj[, 1]), high=unname(ci_obj[, 3]))
    
    if (exists("line_data")){
      
      line_data <- line_data %>% 
        bind_rows(line_data_group)  
      ci_data <- ci_data %>%
        bind_rows(ci_data_group)
    } else {
      line_data <- line_data_group
      ci_data <- ci_data_group
    }
  }

  line_data <- line_data %>%
    mutate(source = factor(source, levels=names))
  ci_data <- ci_data %>%
    mutate(source = factor(source, levels=names))
    
  
  g <- ggplot(line_data, aes(x=specificity, color=source))+
    geom_line(aes(y=sensitivity), na.rm=TRUE, size=2)+
    scale_color_manual(values=setNames(line_colors,names), labels=setNames(names, line_colors))+
    geom_ribbon(data=ci_data, aes(x=spec, ymin=low, ymax=high, fill=source), alpha=0.1, colour=NA)+
    scale_fill_manual(values=setNames(line_colors,names),
                      labels=setNames(names, line_colors))+
      labs(x = "Specificity", y = "Sensitivity")+
      theme_classic(base_size=28) +
      theme(plot.caption = element_text(hjust = 0,9, vjust=75, size=18),
            legend.title=element_blank(),
            legend.position = c(0.75,0.2)) +
      guides(color=guide_legend(ncol=n_legend_cols))+ 
      scale_x_reverse(breaks = scales::pretty_breaks(n = 5))+
      scale_y_continuous(breaks = scales::pretty_breaks(n = 5)) + 
    geom_segment(aes(x=1,y=0,xend=0,yend=1), color="black", lty="longdash")
                      
  plot(g)

  return(g)
}
  
g<- plot_multiple_roc_curve_with_ci(list(rocs_1y$pce_risk,
                                         rocs_1y$img_clin_fusion_preds),
                                    ci_objs_1y_fig,
                               c("PCE", "I+C"),
                               c("black", "blue"),
                               1)

g<- plot_multiple_roc_curve_with_ci(list(rocs_5y$pce_risk,
                                         rocs_5y$img_clin_fusion_preds),
                                    ci_objs_5y_fig,
                                    c("PCE", "I+C"),
                                    c("black", "blue"), 
                                    1)

g<- plot_multiple_roc_curve_with_ci(list(rocs_1y$frs,
                                         rocs_1y$pce_risk,
                                         rocs_1y$seg_risk,
                                         rocs_1y$pce_seg_model_pred,
                                         rocs_1y$img_pred,
                                         rocs_1y$clin_pred,
                                         rocs_1y$img_clin_fusion_preds,
                                         rocs_1y$img_clin_seg_fusion_preds
                                         ),
                                    ci_objs_1y_fig2,
                                    c("FRS", "PCE", "S", "PCE+S", "I", "C", "I+C", "I+C+S"),
                                    c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
                                    )

g<- plot_multiple_roc_curve_with_ci(list(rocs_5y$frs,
                                         rocs_5y$pce_risk,
                                         rocs_5y$seg_risk,
                                         rocs_5y$pce_seg_model_pred,
                                         rocs_5y$img_pred,
                                         rocs_5y$clin_pred,
                                         rocs_5y$img_clin_fusion_preds,
                                         rocs_5y$img_clin_seg_fusion_preds),
                                    ci_objs_5y_fig2,
                                    c("FRS", "PCE", "S", "PCE+S", "I", "C", "I+C", "I+C+S"),
                                    c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7")
                                    )

#Precision recall
# Custom stratified bootstrap for PRC 


get_auprc <- function(precision, recall){
  #based on https://github.com/xrobin/pROC/blob/master/R/auc.R
  diffs.x <- recall[-1] - recall[-length(recall)]
  means.vert <- (precision[-1]+precision[-length(precision)])/2
  auc <- sum(means.vert*diffs.x, na.rm=TRUE)
  return(auc)
}

get_auprc_fromrocobj <- function(rocobj){
  sensitivities <- seq(0, 1, l=25)
  precision <- coords(rocobj, 
                      sensitivities, 
                      input="sens", 
                      ret=c("precision"),
                      transpose=FALSE)
  return(get_auprc(precision, sensitivities))
}

bootstrap_auprc <- function(labels, preds, n.bootstraps=2000, stratified=TRUE){
  # # set up parallelization -> no real performance improvement
  # num_cores <- detectCores() - 1
  # cl <- makeCluster(num_cores, type="FORK")
  
  tbl <- data.frame(labels, preds) %>% 
    as_tibble()
  
  n <- length(labels)
  num_pos <- sum(labels==1)
  
  sensitivities <- seq(0,1,length.out=50)
  
  bootstrap_iter <- function(data=tbl){
    if(stratified){
      sample <- data %>% 
        filter(labels==0) %>% 
        sample_n(n-num_pos, replace=TRUE) %>%
        rbind(sample_n(filter(data, labels==1), num_pos, replace=TRUE))
    }
    else{
      sample <- data %>%
                 sample_n(n, replace=TRUE)
    }
    
    rocObj <- roc(labels ~ preds, sample, quiet=TRUE,auc=FALSE)
    get_prec_rect <- function(recall,roc_object=rocObj){
      return(coords(roc_object, recall, input="sens", ret=c("precision"), transpose=FALSE))
    }
    precisions <- sapply(sensitivities, get_prec_rect) 
    return(precisions)
  }
  prec_rec <- replicate(n.bootstraps, bootstrap_iter(), simplify="matrix")
  return(prec_rec)
}

# plot multiple curves in the same axes
plot_prc_curve_with_ci_vec <- function(roc_objs, bootstraps, names, line_colors) {
  
  sensitivities <- seq(0, 1, l=50)
  default_prev <- sum(roc_objs[[1]]$response==1)/length(roc_objs[[1]]$response)
  
  plt_data <- tibble("sensitivity" = sensitivities)

  for (i in 1:length(roc_objs)){
    precision <- sapply(sensitivities, 
                        function(x) coords(roc_objs[[i]], 
                                           x, 
                                           input="sens", 
                                           ret=c("precision"),
                                           transpose=FALSE))  
    low_high_ci <- apply(bootstraps[[i]], 
                         1, 
                         function(x) quantile(x, prob=c(.025, .975),na.rm=TRUE))
    to_bind <- tibble(prec=precision, low=low_high_ci["2.5%",], high=low_high_ci["97.5%",])
    col_names <- paste(line_colors[i], names(to_bind), sep=".")
    names(to_bind) = col_names
    plt_data <- plt_data %>% 
      bind_cols(to_bind) 

  }

  plt_data<- plt_data %>%
    gather(key, value, -sensitivity) %>%
    separate(key, into=c('color','measurement'), sep='\\.') %>% 
    spread(measurement, value) %>%
    mutate(color=factor(color, levels=line_colors))
  

  g <- ggplot(plt_data, aes(x=sensitivity, group=color, color=color))+
    geom_line(aes(y=prec), na.rm=TRUE, size=2)+
    scale_color_manual(values=line_colors, labels=setNames(names, line_colors))+
    geom_ribbon(aes(ymin=low, ymax=high, fill=color), alpha=0.1, colour=NA)+
    scale_fill_manual(values=line_colors,
                      labels=setNames(names, line_colors))

  line_vals <- setNames(line_colors, line_colors)
  g <- g +
      labs(x = "Recall (Sensitivity)", y = "Precision")+
      theme_classic(base_size=28) +
      theme(plot.caption = element_text(hjust = 0.9, vjust=75, size=18),
           legend.title=element_blank(),
           legend.position = c(.7, 11)) +
    scale_x_continuous(breaks = scales::pretty_breaks(n = 5))+
    scale_y_continuous(breaks = scales::pretty_breaks(n = 5))+
    geom_segment(aes(x=0,y=default_prev,xend=1,yend=default_prev), color="black", lty="longdash", weight=0.5)
  

  plot(g)

  return(g)
}

frs_prc_boot_1y <- bootstrap_auprc(predictions_1y$label, predictions_1y$frs)
pce_prc_boot_1y <- bootstrap_auprc(predictions_1y$label, predictions_1y$pce_risk)
seg_prc_boot_1y <- bootstrap_auprc(predictions_1y$label, predictions_1y$seg_risk)
pce_seg_prc_boot_1y <- bootstrap_auprc(predictions_1y$label, predictions_1y$pce_seg_model_pred)
img_prc_boot_1y <- bootstrap_auprc(predictions_1y$label, predictions_1y$img_pred)
clin_prc_boot_1y <- bootstrap_auprc(predictions_1y$label, predictions_1y$clin_pred)
fusion_prc_boot_1y <- bootstrap_auprc(predictions_1y$label, predictions_1y$img_clin_fusion_preds)
fusion2_prc_boot_1y <- bootstrap_auprc(predictions_1y$label, predictions_1y$img_clin_seg_fusion_preds)




frs_prc_boot_5y <- bootstrap_auprc(predictions_5y$label, predictions_5y$frs)
pce_prc_boot_5y <- bootstrap_auprc(predictions_5y$label, predictions_5y$pce_risk)
seg_prc_boot_5y <- bootstrap_auprc(predictions_5y$label, predictions_5y$seg_risk)
pce_seg_prc_boot_5y <- bootstrap_auprc(predictions_5y$label, predictions_5y$pce_seg_model_pred)
img_prc_boot_5y <- bootstrap_auprc(predictions_5y$label, predictions_5y$img_pred)
clin_prc_boot_5y <- bootstrap_auprc(predictions_5y$label, predictions_5y$clin_pred)
fusion_prc_boot_5y <- bootstrap_auprc(predictions_5y$label, predictions_5y$img_clin_fusion_preds)
fusion2_prc_boot_5y <- bootstrap_auprc(predictions_5y$label, predictions_5y$img_clin_seg_fusion_preds)


g<- plot_prc_curve_with_ci_vec(list(roc(predictions_1y$label, predictions_1y$pce_risk, auc=FALSE, quiet=TRUE),
                                    roc(predictions_1y$label, predictions_1y$img_clin_fusion_preds, auc=FALSE, quiet=TRUE)),
                               list(pce_prc_boot_1y, fusion_prc_boot_1y),
                               c("PCE", "Imaging + Clinical Fusion"),
                               c("black", "blue"))

g<- plot_prc_curve_with_ci_vec(list(roc(predictions_5y$label, predictions_5y$pce_risk, auc=FALSE, quiet=TRUE),
                                    roc(predictions_5y$label, predictions_5y$img_clin_fusion_preds, auc=FALSE, quiet=TRUE)),
                               list(pce_prc_boot_5y, fusion_prc_boot_5y),
                               c("PCE", "Imaging + Clinical Fusion"),
                               c("black", "blue"))


g<- plot_prc_curve_with_ci_vec(list(roc(predictions_1y$label, predictions_1y$frs, auc=FALSE, quiet=TRUE),
                                    roc(predictions_1y$label, predictions_1y$pce_risk, auc=FALSE, quiet=TRUE),
                                    roc(predictions_1y$label, predictions_1y$seg_risk, auc=FALSE, quiet=TRUE),
                                    roc(predictions_1y$label, predictions_1y$pce_seg_model_pred, auc=FALSE, quiet=TRUE),
                                    roc(predictions_1y$label, predictions_1y$img_pred, auc=FALSE, quiet=TRUE),
                                    roc(predictions_1y$label, predictions_1y$clin_pred, auc=FALSE, quiet=TRUE),
                                    roc(predictions_1y$label, predictions_1y$img_clin_fusion_preds, auc=FALSE, quiet=TRUE),
                                    roc(predictions_1y$label, predictions_1y$img_clin_seg_fusion_preds, auc=FALSE, quiet=TRUE)),
                               list(frs_prc_boot_1y, pce_prc_boot_1y, seg_prc_boot_1y, 
                                    pce_seg_prc_boot_1y, img_prc_boot_1y,clin_prc_boot_1y,
                                    fusion_prc_boot_1y, fusion2_prc_boot_1y),
                               c("FRS", "PCE", "S", "PCE+S", "I", "C", "I+C", "I+C+S"),
                               c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))

g<- plot_prc_curve_with_ci_vec(list(roc(predictions_5y$label, predictions_5y$frs, auc=FALSE, quiet=TRUE),
                                    roc(predictions_5y$label, predictions_5y$pce_risk, auc=FALSE, quiet=TRUE),
                                    roc(predictions_5y$label, predictions_5y$seg_risk, auc=FALSE, quiet=TRUE),
                                    roc(predictions_5y$label, predictions_5y$pce_seg_model_pred, auc=FALSE, quiet=TRUE),
                                    roc(predictions_5y$label, predictions_5y$img_pred, auc=FALSE, quiet=TRUE),
                                    roc(predictions_5y$label, predictions_5y$clin_pred, auc=FALSE, quiet=TRUE),
                                    roc(predictions_5y$label, predictions_5y$img_clin_fusion_preds, auc=FALSE, quiet=TRUE),
                                    roc(predictions_5y$label, predictions_5y$img_clin_seg_fusion_preds, auc=FALSE, quiet=TRUE)),
                               list(frs_prc_boot_5y, pce_prc_boot_5y, seg_prc_boot_5y, 
                                    pce_seg_prc_boot_5y, img_prc_boot_5y,clin_prc_boot_5y,
                                    fusion_prc_boot_5y, fusion2_prc_boot_5y),
                               c("FRS", "PCE", "S", "PCE+S", "I", "C", "I+C", "I+C+S"),
                               c("#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"))
