#media ponderata esami
import numpy as np

esami_cfu = [[22, 6], [25, 9], [25, 9],[25, 9], [27, 9], [22, 12], [19, 3], [27, 12], [27, 6], [26, 12], [30, 6], [27, 12], [27, 9], [27, 9],[25, 6], [28, 6], [27, 9], [27, 6], [26, 12], [28, 9]]
print(len(esami_cfu))
media_ponderata = np.average([esami[0] for esami in esami_cfu], weights=[esami[1] for esami in esami_cfu])
print(f"Media ponderata degli esami: {media_ponderata:.2f}")

esami_cfu_magistrale = [[28, 9], [27, 9], [27, 9], [25, 9], [28, 9], [26, 9], [28, 9], [29, 9], [30, 3]]
media_ponderata_magistrale = np.average([esami[0] for esami in esami_cfu_magistrale], weights=[esami[1] for esami in esami_cfu_magistrale])
print(f"Media ponderata degli esami magistrali: {media_ponderata_magistrale:.2f}")