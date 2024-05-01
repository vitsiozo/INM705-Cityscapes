s/\\crefrangeconjunction/ to\\nobreakspace /g 
s/\\crefrangepreconjunction//g 
s/\\crefrangepostconjunction//g 
s/\\crefpairconjunction/ and\\nobreakspace /g 
s/\\crefmiddleconjunction/, /g 
s/\\creflastconjunction/ and\\nobreakspace /g 
s/\\crefpairgroupconjunction/ and\\nobreakspace /g 
s/\\crefmiddlegroupconjunction/, /g 
s/\\creflastgroupconjunction/, and\\nobreakspace /g 
s/\\cref@line@name /Line/g 
s/\\cref@line@name@plural /Lines/g 
s/\\Cref@line@name /Line/g 
s/\\Cref@line@name@plural /Lines/g 
s/\\cref@listing@name /Listing/g 
s/\\cref@listing@name@plural /Listings/g 
s/\\Cref@listing@name /Listing/g 
s/\\Cref@listing@name@plural /Listings/g 
s/\\cref@algorithm@name /Algorithm/g 
s/\\cref@algorithm@name@plural /Algorithms/g 
s/\\Cref@algorithm@name /Algorithm/g 
s/\\Cref@algorithm@name@plural /Algorithms/g 
s/\\cref@note@name /Note/g 
s/\\cref@note@name@plural /Notes/g 
s/\\Cref@note@name /Note/g 
s/\\Cref@note@name@plural /Notes/g 
s/\\cref@remark@name /Remark/g 
s/\\cref@remark@name@plural /Remarks/g 
s/\\Cref@remark@name /Remark/g 
s/\\Cref@remark@name@plural /Remarks/g 
s/\\cref@example@name /Example/g 
s/\\cref@example@name@plural /Examples/g 
s/\\Cref@example@name /Example/g 
s/\\Cref@example@name@plural /Examples/g 
s/\\cref@result@name /Result/g 
s/\\cref@result@name@plural /Results/g 
s/\\Cref@result@name /Result/g 
s/\\Cref@result@name@plural /Results/g 
s/\\cref@definition@name /Definition/g 
s/\\cref@definition@name@plural /Definitions/g 
s/\\Cref@definition@name /Definition/g 
s/\\Cref@definition@name@plural /Definitions/g 
s/\\cref@proposition@name /Proposition/g 
s/\\cref@proposition@name@plural /Propositions/g 
s/\\Cref@proposition@name /Proposition/g 
s/\\Cref@proposition@name@plural /Propositions/g 
s/\\cref@corollary@name /Corollary/g 
s/\\cref@corollary@name@plural /Corollaries/g 
s/\\Cref@corollary@name /Corollary/g 
s/\\Cref@corollary@name@plural /Corollaries/g 
s/\\cref@lemma@name /Lemma/g 
s/\\cref@lemma@name@plural /Lemmas/g 
s/\\Cref@lemma@name /Lemma/g 
s/\\Cref@lemma@name@plural /Lemmas/g 
s/\\cref@theorem@name /Theorem/g 
s/\\cref@theorem@name@plural /Theorems/g 
s/\\Cref@theorem@name /Theorem/g 
s/\\Cref@theorem@name@plural /Theorems/g 
s/\\cref@footnote@name /Footnote/g 
s/\\cref@footnote@name@plural /Footnotes/g 
s/\\Cref@footnote@name /Footnote/g 
s/\\Cref@footnote@name@plural /Footnotes/g 
s/\\cref@enumi@name /Item/g 
s/\\cref@enumi@name@plural /Items/g 
s/\\Cref@enumi@name /Item/g 
s/\\Cref@enumi@name@plural /Items/g 
s/\\cref@appendix@name /Appendix/g 
s/\\cref@appendix@name@plural /Appendices/g 
s/\\Cref@appendix@name /Appendix/g 
s/\\Cref@appendix@name@plural /Appendices/g 
s/\\cref@section@name /Section/g 
s/\\cref@section@name@plural /Sections/g 
s/\\Cref@section@name /Section/g 
s/\\Cref@section@name@plural /Sections/g 
s/\\cref@chapter@name /Chapter/g 
s/\\cref@chapter@name@plural /Chapters/g 
s/\\Cref@chapter@name /Chapter/g 
s/\\Cref@chapter@name@plural /Chapters/g 
s/\\cref@part@name /Part/g 
s/\\cref@part@name@plural /Parts/g 
s/\\Cref@part@name /Part/g 
s/\\Cref@part@name@plural /Parts/g 
s/\\cref@page@name /Page/g 
s/\\cref@page@name@plural /Pages/g 
s/\\Cref@page@name /Page/g 
s/\\Cref@page@name@plural /Pages/g 
s/\\cref@table@name /Table/g 
s/\\cref@table@name@plural /Tables/g 
s/\\Cref@table@name /Table/g 
s/\\Cref@table@name@plural /Tables/g 
s/\\cref@figure@name /Figure/g 
s/\\cref@figure@name@plural /Figures/g 
s/\\Cref@figure@name /Figure/g 
s/\\Cref@figure@name@plural /Figures/g 
s/\\cref@equation@name /Equation/g 
s/\\cref@equation@name@plural /Equations/g 
s/\\Cref@equation@name /Equation/g 
s/\\Cref@equation@name@plural /Equations/g 
s/\\label\[[^]]*\]/\\label/g
s/\\usepackage\(\[.*\]\)\{0,1\}{cleveref}//g
s/\\[cC]refformat{.*}{.*}//g
s/\\[cC]refrangeformat{.*}{.*}//g
s/\\[cC]refmultiformat{.*}{.*}{.*}{.*}//g
s/\\[cC]refrangemultiformat{.*}{.*}{.*}{.*}//g
s/\\[cC]refname{.*}{.*}//g
s/\\[cC]reflabelformat{.*}{.*}//g
s/\\[cC]refrangelabelformat{.*}{.*}//g
s/\\[cC]refdefaultlabelformat{.*}//g
s/\\renewcommand{\\crefpairconjunction}{.*}//g
s/\\renewcommand{\\crefpairgroupconjunction}{.*}//g
s/\\renewcommand{\\crefmiddleconjunction}{.*}//g
s/\\renewcommand{\\crefmiddlegroupconjunction}{.*}//g
s/\\renewcommand{\\creflastconjunction}{.*}//g
s/\\renewcommand{\\creflastgroupconjunction}{.*}//g
s/\\renewcommand{\\[cC]ref}{.*}//g
s/\\renewcommand{\\[cC]refrange}{.*}//g

