# some thoughts aboug meta learning

$$
\theta^* = \arg \min_{\theta}\;\lambda \cdot \frac{1}{m}\sum_{i=1}^{m}KL(y_i^v||f(x_i^v;w^*(\theta)))+(1-\lambda)\cdot \big[ \frac{1}{m}\sum_{i=1}^{m}\big(-\log p_{\theta}^{y_i^v}(\text{IC}|x_i^v,y_i^v) -\frac{1}{C-1}\sum_{j=1\dots C,j\neq y_i^v}\log p_{\theta}^j(\text{NIC}|x_i^v,y_i^v)\big)\big]\\
s.t.\quad w^*(\theta)=\arg \min_{w} \mathbb{E}_{(x,\tilde{y})\,\sim \,\tilde{\mathcal{D}}}\big[
p_{\theta}^{\tilde{y}}(\text{IC}|x,\tilde{y})\cdot \mathcal{l}_{CE}(\tilde{y},f(x;w)) \big]\\
$$

$$
\theta^* = \arg \min_{\theta}\;\mathbb{E}_{(x^v,y^v)\sim D_v}\big[-\log g_{\theta}(f(x^v,w^*),y^v)-\log (1-g_{\theta}(f(x^v,w^*),y^{v,\text{hard}}))-\log (1-g_{\theta}(f(x^v,w^*),y^{v,\text{rand}}))\big]\\
s.t.\quad w^*(\theta)=\arg \min_{w} \mathbb{E}_{(x,\tilde{y})\,\sim \,\tilde{\mathcal{D}}}\big[
g_{\theta}(f(x;w),\tilde{y})\cdot \mathcal{l}_{CE}(\tilde{y},f(x;w)) \big]\\
y^{v,\text{hard}} = \arg \max_{j\neq y^v}f_j(x^v;w^*)\\
y^{v,\text{rand}} = \arg \max_{j\neq y^v,y^{v,\text{hard}}}f_j(x^v;w^*))
$$

$$
w^* = \arg \min_{w}\;\mathbb{E}_{(x^v,y^v)\sim D^v}\big[l_{ce}(g_1(f(x^v);\theta^*),y^v)\big]\\
s.t.\quad \theta^*(w) = \arg \min_{\theta}\;\mathbb{E}_{(x,\tilde{y})\sim \tilde{D}}\big[\frac{g_1^{\tilde{y}}}{({T(f(x);w)}^Tg_1)^{\tilde{y}}}l_{ce}(g_1(f(x);\theta),\tilde{y})\big]
$$