import numpy as np
import matplotlib.pyplot as plt

v_1_a = np.arange(-5,-2.725,0.3)
v_2 = np.arange(0,30,0.3)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for v_1 in v_1_a:
    y = (0.21201790869235992+np.tanh(-0.5434480905532837+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.06577640026807785+((v_2-0.0)*2/(30.0-0.0)-1)*-0.034473076462745667+-0.09293169528245926)*0.02008296735584736+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.25829973816871643+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18427953124046326+0.4967438876628876)*0.4247687757015228+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.9875601530075073+((v_2-0.0)*2/(30.0-0.0)-1)*0.12945041060447693+-1.1130180358886719)*0.8321529626846313+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.7927579879760742+((v_2-0.0)*2/(30.0-0.0)-1)*0.29469817876815796+0.45779263973236084)*0.8550235033035278+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.4969322085380554+((v_2-0.0)*2/(30.0-0.0)-1)*0.17769545316696167+-0.7001920342445374)*0.06769903749227524+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.08743591606616974+((v_2-0.0)*2/(30.0-0.0)-1)*-0.09620935469865799+0.2640410363674164)*0.07777582854032516+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.118822693824768+((v_2-0.0)*2/(30.0-0.0)-1)*2.1054787635803223+0.10783213376998901)*-0.039218928664922714+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.0931097269058228+((v_2-0.0)*2/(30.0-0.0)-1)*1.9066548347473145+0.12152523547410965)*0.13402636349201202+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.5400255918502808+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18525753915309906+1.098791480064392)*-1.2051098346710205+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.8753900527954102+((v_2-0.0)*2/(30.0-0.0)-1)*-0.015145407058298588+0.8246061205863953)*-0.010737224481999874+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.0486735105514526+((v_2-0.0)*2/(30.0-0.0)-1)*-0.16050373017787933+1.2661125659942627)*-1.2151411771774292+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.9385932087898254+((v_2-0.0)*2/(30.0-0.0)-1)*1.0998767614364624+0.21459171175956726)*0.17192694544792175+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.138947606086731+((v_2-0.0)*2/(30.0-0.0)-1)*-0.08288687467575073+1.1685595512390137)*0.15703456103801727+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.46504345536231995+((v_2-0.0)*2/(30.0-0.0)-1)*-2.5573036670684814+-1.254887342453003)*0.4874262809753418+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.27722007036209106+((v_2-0.0)*2/(30.0-0.0)-1)*0.621958315372467+-0.027641907334327698)*-1.262939214706421)*-0.7160813808441162+np.tanh(0.36592555046081543+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.06577640026807785+((v_2-0.0)*2/(30.0-0.0)-1)*-0.034473076462745667+-0.09293169528245926)*0.0022610430605709553+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.25829973816871643+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18427953124046326+0.4967438876628876)*0.04531489685177803+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.9875601530075073+((v_2-0.0)*2/(30.0-0.0)-1)*0.12945041060447693+-1.1130180358886719)*-1.5979737043380737+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.7927579879760742+((v_2-0.0)*2/(30.0-0.0)-1)*0.29469817876815796+0.45779263973236084)*0.16437870264053345+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.4969322085380554+((v_2-0.0)*2/(30.0-0.0)-1)*0.17769545316696167+-0.7001920342445374)*-0.06634145230054855+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.08743591606616974+((v_2-0.0)*2/(30.0-0.0)-1)*-0.09620935469865799+0.2640410363674164)*-0.005630354396998882+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.118822693824768+((v_2-0.0)*2/(30.0-0.0)-1)*2.1054787635803223+0.10783213376998901)*-0.8086069226264954+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.0931097269058228+((v_2-0.0)*2/(30.0-0.0)-1)*1.9066548347473145+0.12152523547410965)*-0.8344175815582275+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.5400255918502808+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18525753915309906+1.098791480064392)*0.8022366762161255+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.8753900527954102+((v_2-0.0)*2/(30.0-0.0)-1)*-0.015145407058298588+0.8246061205863953)*0.7487179636955261+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.0486735105514526+((v_2-0.0)*2/(30.0-0.0)-1)*-0.16050373017787933+1.2661125659942627)*1.3573427200317383+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.9385932087898254+((v_2-0.0)*2/(30.0-0.0)-1)*1.0998767614364624+0.21459171175956726)*0.5858700275421143+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.138947606086731+((v_2-0.0)*2/(30.0-0.0)-1)*-0.08288687467575073+1.1685595512390137)*3.113992691040039+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.46504345536231995+((v_2-0.0)*2/(30.0-0.0)-1)*-2.5573036670684814+-1.254887342453003)*-0.6780874729156494+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.27722007036209106+((v_2-0.0)*2/(30.0-0.0)-1)*0.621958315372467+-0.027641907334327698)*0.2686059772968292)*-0.8205653429031372+np.tanh(-0.6022824048995972+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.06577640026807785+((v_2-0.0)*2/(30.0-0.0)-1)*-0.034473076462745667+-0.09293169528245926)*0.18121226131916046+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.25829973816871643+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18427953124046326+0.4967438876628876)*-0.6131199598312378+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.9875601530075073+((v_2-0.0)*2/(30.0-0.0)-1)*0.12945041060447693+-1.1130180358886719)*1.8493021726608276+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.7927579879760742+((v_2-0.0)*2/(30.0-0.0)-1)*0.29469817876815796+0.45779263973236084)*0.09615162014961243+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.4969322085380554+((v_2-0.0)*2/(30.0-0.0)-1)*0.17769545316696167+-0.7001920342445374)*0.9289898872375488+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.08743591606616974+((v_2-0.0)*2/(30.0-0.0)-1)*-0.09620935469865799+0.2640410363674164)*-0.5591931939125061+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.118822693824768+((v_2-0.0)*2/(30.0-0.0)-1)*2.1054787635803223+0.10783213376998901)*2.685056447982788+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.0931097269058228+((v_2-0.0)*2/(30.0-0.0)-1)*1.9066548347473145+0.12152523547410965)*2.6047651767730713+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.5400255918502808+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18525753915309906+1.098791480064392)*-0.724210262298584+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.8753900527954102+((v_2-0.0)*2/(30.0-0.0)-1)*-0.015145407058298588+0.8246061205863953)*-0.833300769329071+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.0486735105514526+((v_2-0.0)*2/(30.0-0.0)-1)*-0.16050373017787933+1.2661125659942627)*-1.3105493783950806+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.9385932087898254+((v_2-0.0)*2/(30.0-0.0)-1)*1.0998767614364624+0.21459171175956726)*-0.5814719796180725+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.138947606086731+((v_2-0.0)*2/(30.0-0.0)-1)*-0.08288687467575073+1.1685595512390137)*-3.3401763439178467+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.46504345536231995+((v_2-0.0)*2/(30.0-0.0)-1)*-2.5573036670684814+-1.254887342453003)*1.695241093635559+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.27722007036209106+((v_2-0.0)*2/(30.0-0.0)-1)*0.621958315372467+-0.027641907334327698)*0.17898063361644745)*0.24068714678287506+np.tanh(0.10625965893268585+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.06577640026807785+((v_2-0.0)*2/(30.0-0.0)-1)*-0.034473076462745667+-0.09293169528245926)*-0.2162550836801529+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.25829973816871643+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18427953124046326+0.4967438876628876)*0.7282456159591675+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.9875601530075073+((v_2-0.0)*2/(30.0-0.0)-1)*0.12945041060447693+-1.1130180358886719)*-1.0391075611114502+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.7927579879760742+((v_2-0.0)*2/(30.0-0.0)-1)*0.29469817876815796+0.45779263973236084)*1.0106048583984375+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.4969322085380554+((v_2-0.0)*2/(30.0-0.0)-1)*0.17769545316696167+-0.7001920342445374)*-0.993887186050415+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.08743591606616974+((v_2-0.0)*2/(30.0-0.0)-1)*-0.09620935469865799+0.2640410363674164)*0.6245580911636353+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.118822693824768+((v_2-0.0)*2/(30.0-0.0)-1)*2.1054787635803223+0.10783213376998901)*-1.2793474197387695+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.0931097269058228+((v_2-0.0)*2/(30.0-0.0)-1)*1.9066548347473145+0.12152523547410965)*-1.381887674331665+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.5400255918502808+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18525753915309906+1.098791480064392)*0.2308318167924881+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.8753900527954102+((v_2-0.0)*2/(30.0-0.0)-1)*-0.015145407058298588+0.8246061205863953)*4.086181163787842+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.0486735105514526+((v_2-0.0)*2/(30.0-0.0)-1)*-0.16050373017787933+1.2661125659942627)*0.3922869861125946+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.9385932087898254+((v_2-0.0)*2/(30.0-0.0)-1)*1.0998767614364624+0.21459171175956726)*-0.380303293466568+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.138947606086731+((v_2-0.0)*2/(30.0-0.0)-1)*-0.08288687467575073+1.1685595512390137)*2.493921995162964+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.46504345536231995+((v_2-0.0)*2/(30.0-0.0)-1)*-2.5573036670684814+-1.254887342453003)*0.07887153327465057+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.27722007036209106+((v_2-0.0)*2/(30.0-0.0)-1)*0.621958315372467+-0.027641907334327698)*0.20903140306472778)*0.22692926228046417+np.tanh(0.13732905685901642+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.06577640026807785+((v_2-0.0)*2/(30.0-0.0)-1)*-0.034473076462745667+-0.09293169528245926)*-0.12640607357025146+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.25829973816871643+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18427953124046326+0.4967438876628876)*-0.45985978841781616+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.9875601530075073+((v_2-0.0)*2/(30.0-0.0)-1)*0.12945041060447693+-1.1130180358886719)*0.1837712824344635+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.7927579879760742+((v_2-0.0)*2/(30.0-0.0)-1)*0.29469817876815796+0.45779263973236084)*0.012326812371611595+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.4969322085380554+((v_2-0.0)*2/(30.0-0.0)-1)*0.17769545316696167+-0.7001920342445374)*-0.4783305525779724+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.08743591606616974+((v_2-0.0)*2/(30.0-0.0)-1)*-0.09620935469865799+0.2640410363674164)*0.24770142138004303+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.118822693824768+((v_2-0.0)*2/(30.0-0.0)-1)*2.1054787635803223+0.10783213376998901)*-0.2932775020599365+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.0931097269058228+((v_2-0.0)*2/(30.0-0.0)-1)*1.9066548347473145+0.12152523547410965)*0.24666708707809448+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.5400255918502808+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18525753915309906+1.098791480064392)*-0.3352695107460022+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.8753900527954102+((v_2-0.0)*2/(30.0-0.0)-1)*-0.015145407058298588+0.8246061205863953)*0.4531196355819702+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.0486735105514526+((v_2-0.0)*2/(30.0-0.0)-1)*-0.16050373017787933+1.2661125659942627)*-0.0810389295220375+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.9385932087898254+((v_2-0.0)*2/(30.0-0.0)-1)*1.0998767614364624+0.21459171175956726)*-0.11758299171924591+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.138947606086731+((v_2-0.0)*2/(30.0-0.0)-1)*-0.08288687467575073+1.1685595512390137)*-0.12323170900344849+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.46504345536231995+((v_2-0.0)*2/(30.0-0.0)-1)*-2.5573036670684814+-1.254887342453003)*0.0008708505192771554+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.27722007036209106+((v_2-0.0)*2/(30.0-0.0)-1)*0.621958315372467+-0.027641907334327698)*0.1448609083890915)*0.00370961450971663+np.tanh(0.19637656211853027+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.06577640026807785+((v_2-0.0)*2/(30.0-0.0)-1)*-0.034473076462745667+-0.09293169528245926)*-0.019097821786999702+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.25829973816871643+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18427953124046326+0.4967438876628876)*-0.4575819671154022+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.9875601530075073+((v_2-0.0)*2/(30.0-0.0)-1)*0.12945041060447693+-1.1130180358886719)*0.11380714923143387+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.7927579879760742+((v_2-0.0)*2/(30.0-0.0)-1)*0.29469817876815796+0.45779263973236084)*0.611789882183075+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.4969322085380554+((v_2-0.0)*2/(30.0-0.0)-1)*0.17769545316696167+-0.7001920342445374)*0.49024197459220886+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.08743591606616974+((v_2-0.0)*2/(30.0-0.0)-1)*-0.09620935469865799+0.2640410363674164)*0.04788047820329666+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.118822693824768+((v_2-0.0)*2/(30.0-0.0)-1)*2.1054787635803223+0.10783213376998901)*-0.13709545135498047+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.0931097269058228+((v_2-0.0)*2/(30.0-0.0)-1)*1.9066548347473145+0.12152523547410965)*0.09547130018472672+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.5400255918502808+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18525753915309906+1.098791480064392)*0.08220274746417999+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.8753900527954102+((v_2-0.0)*2/(30.0-0.0)-1)*-0.015145407058298588+0.8246061205863953)*0.686958909034729+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.0486735105514526+((v_2-0.0)*2/(30.0-0.0)-1)*-0.16050373017787933+1.2661125659942627)*-0.19031868875026703+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.9385932087898254+((v_2-0.0)*2/(30.0-0.0)-1)*1.0998767614364624+0.21459171175956726)*-0.2795042395591736+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.138947606086731+((v_2-0.0)*2/(30.0-0.0)-1)*-0.08288687467575073+1.1685595512390137)*-0.5566036701202393+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.46504345536231995+((v_2-0.0)*2/(30.0-0.0)-1)*-2.5573036670684814+-1.254887342453003)*0.03902744501829147+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.27722007036209106+((v_2-0.0)*2/(30.0-0.0)-1)*0.621958315372467+-0.027641907334327698)*-0.1364305466413498)*-0.11732948571443558+np.tanh(0.19903317093849182+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.06577640026807785+((v_2-0.0)*2/(30.0-0.0)-1)*-0.034473076462745667+-0.09293169528245926)*-0.1076565757393837+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.25829973816871643+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18427953124046326+0.4967438876628876)*-0.2780713737010956+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.9875601530075073+((v_2-0.0)*2/(30.0-0.0)-1)*0.12945041060447693+-1.1130180358886719)*0.20731189846992493+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.7927579879760742+((v_2-0.0)*2/(30.0-0.0)-1)*0.29469817876815796+0.45779263973236084)*0.14668411016464233+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.4969322085380554+((v_2-0.0)*2/(30.0-0.0)-1)*0.17769545316696167+-0.7001920342445374)*0.31888291239738464+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.08743591606616974+((v_2-0.0)*2/(30.0-0.0)-1)*-0.09620935469865799+0.2640410363674164)*-0.20303533971309662+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.118822693824768+((v_2-0.0)*2/(30.0-0.0)-1)*2.1054787635803223+0.10783213376998901)*0.29430273175239563+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.0931097269058228+((v_2-0.0)*2/(30.0-0.0)-1)*1.9066548347473145+0.12152523547410965)*-0.2670036554336548+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.5400255918502808+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18525753915309906+1.098791480064392)*-0.11457450687885284+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.8753900527954102+((v_2-0.0)*2/(30.0-0.0)-1)*-0.015145407058298588+0.8246061205863953)*-0.4114856421947479+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.0486735105514526+((v_2-0.0)*2/(30.0-0.0)-1)*-0.16050373017787933+1.2661125659942627)*0.5950798988342285+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.9385932087898254+((v_2-0.0)*2/(30.0-0.0)-1)*1.0998767614364624+0.21459171175956726)*0.0188194140791893+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.138947606086731+((v_2-0.0)*2/(30.0-0.0)-1)*-0.08288687467575073+1.1685595512390137)*0.16134296357631683+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.46504345536231995+((v_2-0.0)*2/(30.0-0.0)-1)*-2.5573036670684814+-1.254887342453003)*-0.001951403683051467+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.27722007036209106+((v_2-0.0)*2/(30.0-0.0)-1)*0.621958315372467+-0.027641907334327698)*-0.34913164377212524)*0.01688460446894169+np.tanh(0.0838526114821434+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.06577640026807785+((v_2-0.0)*2/(30.0-0.0)-1)*-0.034473076462745667+-0.09293169528245926)*0.19114157557487488+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.25829973816871643+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18427953124046326+0.4967438876628876)*-0.20413725078105927+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.9875601530075073+((v_2-0.0)*2/(30.0-0.0)-1)*0.12945041060447693+-1.1130180358886719)*0.46137890219688416+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.7927579879760742+((v_2-0.0)*2/(30.0-0.0)-1)*0.29469817876815796+0.45779263973236084)*0.0466080978512764+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.4969322085380554+((v_2-0.0)*2/(30.0-0.0)-1)*0.17769545316696167+-0.7001920342445374)*0.16111232340335846+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.08743591606616974+((v_2-0.0)*2/(30.0-0.0)-1)*-0.09620935469865799+0.2640410363674164)*0.0006455738330259919+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.118822693824768+((v_2-0.0)*2/(30.0-0.0)-1)*2.1054787635803223+0.10783213376998901)*-0.17206573486328125+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*1.0931097269058228+((v_2-0.0)*2/(30.0-0.0)-1)*1.9066548347473145+0.12152523547410965)*0.10363622009754181+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.5400255918502808+((v_2-0.0)*2/(30.0-0.0)-1)*-0.18525753915309906+1.098791480064392)*0.16633127629756927+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.8753900527954102+((v_2-0.0)*2/(30.0-0.0)-1)*-0.015145407058298588+0.8246061205863953)*0.6608903408050537+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.0486735105514526+((v_2-0.0)*2/(30.0-0.0)-1)*-0.16050373017787933+1.2661125659942627)*-0.21595904231071472+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-0.9385932087898254+((v_2-0.0)*2/(30.0-0.0)-1)*1.0998767614364624+0.21459171175956726)*-0.15759550034999847+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*-1.138947606086731+((v_2-0.0)*2/(30.0-0.0)-1)*-0.08288687467575073+1.1685595512390137)*0.07341654598712921+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.46504345536231995+((v_2-0.0)*2/(30.0-0.0)-1)*-2.5573036670684814+-1.254887342453003)*-0.03501413017511368+np.tanh(((v_1--5.0)*2/(-2.725--5.0)-1)*0.27722007036209106+((v_2-0.0)*2/(30.0-0.0)-1)*0.621958315372467+-0.027641907334327698)*0.047894224524497986)*-0.005665074102580547)


# # Plot against model.predict to verify generated function
# v_1 = np.arange(-1,1,0.1)
# v_2 = np.arange(-1,1,0.1)
# for v_1_x in v_1:
#     for v_2_x in v_2:
#         input= [[v_1_x, v_2_x]]
#         output = model.predict(input)
#         plt.scatter(v_2_x, output[0][0])


    # Plot the data

    ax.scatter(v_1, v_2, y , color='g')
plt.show()
