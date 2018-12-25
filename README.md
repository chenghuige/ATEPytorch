# AET(Aspect Extraction Task) Experiments with Pytorch

1. a rather simple implementation of ABAE(ACL-2017-An Unsupervised Neural Attention Model for Aspect Extraction) model with Pytorch platform.

## Environment

* pytorch: 0.4.1
* python: 3.6

## Dataset

**All the dataset are in csv format.**

For example:

1. citysearch dataset train data (first 2 line)

```text
What do I like about Jeollado? I like the 2 for 1 rolls (sometimes 3 for 1) the prices and the variety on the menu
What don't I like? The rolls are tiny so you have to order more anyway and they will often get your order wrong if you stray from the menu
```
2. citysearch dataset test data (first 2 line)

```text
Food,Always a fun place ... the food is deeelish !
Staff,The staff is n't the friendliest or most competent , and I am stickler for service , but everything else about this place makes up for it .
```

## Results

1. ABAE

```text
>>> aspect words (top 20)
	-aspect1: waiter | waitor | scowled | tugged | waitress | shopkeeper | glared | whined | glanced | excitedly | grunted | lectured | hissed | berated | approached | hosed | slaped | sternly | groped | exclaimed |
	-aspect2: didn | wouldn | couldn | dodnt | wasn | didint | hasn | doesn | dint | shouldn | hadnt | iwanted | shoud | idid | breaka | inever | weren | iwould | neverr | immigrated |
	-aspect3: agua | poco | helado | gente | medio | mano | mucho | hace | todo | tapa | una | coca | asi | nada | por | dulce | adoro | alta | esa | causa |
	-aspect4: multicolored | geometric | jeweled | decorative | ornate | lighted | glassware | centerpiece | customizable | foldable | sculptural | textured | stemware | nautical | embellished | kitchenware | doily | handcrafted | recessed | garish |
	-aspect5: menu | dessert | appetizer | buffet | sushi | brunch | fajitas | paella | restaurante | waiter | ceviche | fastfood | steak | risotto | sashimi | waitress | yumm | carbonara | desayuno | restaurant |
	-aspect6: garlic | crusted | roasted | spinach | sauteed | zucchini | teriyaki | grilled | marinated | crispy | tilapia | asparagus | tenderloin | coleslaw | toasted | cashew | jalapeno | shrimp | fennel | parmesan |
	-aspect7: bai | smac | cassi | kura | crampy | momos | maki | shere | tei | gau | lae | harumi | seta | gaijin | condescension | desparate | knowlege | haloween | ber | interupt |
	-aspect8: garder | perdu | comme | garde | gueule | fois | juste | toute | pour | pris | passe | deux | cours | place | prochaine | moi | vie | chanson | faire | soir |
	-aspect9: service | operator | customer | maintenance | client | costumer | sevice | representative | personnel | svc | billing | software | transport | specialist | installation | delivery | engineer | supervisor | assistance | servicing |
	-aspect10: messed | horrendously | severly | completey | sortof | verging | atrociously | lowpoint | horribly | miscounted | majorly | massively | piqued | hideously | higly | greediness | physicaly | bastardized | notgood | contemplated |
	-aspect11: philippe | alexandre | complimenti | martine | periodista | riccardo | christine | ballesteros | silverman | sorrentino | luca | magnifique | coppola | incroyable | bienvenue | ruth | catherine | salut | alum | christophe |
	-aspect12: didn | greaaat | amaaazing | amaaaaazing | awfull | suprisingly | couldn | wouldn | muggy | yucky | greaaaaaaat | ambience | kindof | goooooooood | summery | tastey | wondeful | disheartening | grreat | hasn |
	-aspect13: sheraton | marriott | restaurant | eatery | bistro | hyatt | tavern | waterfront | residence | chinatown | terrace | astoria | manhattan | marriot | midtown | courtyard | headquarters | resturant | soho | hotel |
	-aspect14: knowledgeable | highly | compassionate | competent | attentive | intuitive | imaginative | enthusiastic | expansive | insightful | articulate | indicative | intellect | regarded | knowledgable | rational | hugely | compelling | impactful | immensely |
```