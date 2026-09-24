"""Numbers reported in the paper, for side-by-side comparison in notebooks 08 and 09.

Source: H. Türeli, "CHORD: Comparing HMM and Constant-Q Representation for Decoding Harmony", 2026,
Table I (RWC-R, 15 tracks), Table II and §IV-B (RWC-P, 100 tracks).
"""

import pandas as pd

_TABLE_I = {
    #            Template  P       R       F1     |  HMM  P       R       F1
    "RWC_R001": (0.5364, 0.5465, 0.5414, 0.5861, 0.5971, 0.5915),
    "RWC_R002": (0.5671, 0.6373, 0.6002, 0.6635, 0.7457, 0.7022),
    "RWC_R003": (0.6080, 0.6187, 0.6133, 0.7210, 0.7337, 0.7273),
    "RWC_R004": (0.5286, 0.5365, 0.5325, 0.6241, 0.6334, 0.6287),
    "RWC_R005": (0.4392, 0.4731, 0.4555, 0.4783, 0.5152, 0.4961),
    "RWC_R006": (0.3600, 0.3940, 0.3762, 0.3788, 0.4146, 0.3959),
    "RWC_R007": (0.6127, 0.6388, 0.6255, 0.7731, 0.8060, 0.7892),
    "RWC_R008": (0.3361, 0.3497, 0.3427, 0.4528, 0.4712, 0.4618),
    "RWC_R009": (0.7622, 0.7622, 0.7622, 0.7974, 0.7974, 0.7974),
    "RWC_R010": (0.5271, 0.5494, 0.5380, 0.6497, 0.6772, 0.6632),
    "RWC_R011": (0.6243, 0.6550, 0.6393, 0.7666, 0.8044, 0.7851),
    "RWC_R012": (0.4887, 0.5111, 0.4996, 0.6231, 0.6517, 0.6371),
    "RWC_R013": (0.8024, 0.8069, 0.8046, 0.9012, 0.9063, 0.9037),
    "RWC_R014": (0.5272, 0.5471, 0.5370, 0.7542, 0.7826, 0.7681),
    "RWC_R015": (0.7684, 0.7684, 0.7684, 0.8094, 0.8094, 0.8094),
}

TABLE_I = pd.DataFrame(_TABLE_I, index=pd.MultiIndex.from_product(
    [["CQT-Template", "CQT-HMM (Viterbi)"], ["P", "R", "F1"]])).T
TABLE_I.loc["Average"] = TABLE_I.mean()

#: Table II: RWC-P tracks where template matching beat the HMM (F1 template, F1 HMM).
TABLE_II = pd.DataFrame({
    "RWC_P092": (0.2405, 0.0954), "RWC_P015": (0.3711, 0.2380), "RWC_P028": (0.3064, 0.1902),
    "RWC_P091": (0.4763, 0.3720), "RWC_P089": (0.3779, 0.2781), "RWC_P031": (0.0730, 0.0000),
}, index=["Template_F1", "HMM_F1"]).T

#: §IV-B averages over the 100 RWC-P tracks.
RWC_P_AVERAGE_F1 = {"CQT-Template": 0.4505, "CQT-HMM (Viterbi)": 0.5566}

#: Other tracks the paper singles out in its failure analysis.
FAILURE_EXAMPLES = {"RWC_P031": "house track with four-on-the-floor kick drum; HMM F1 = 0.00",
                    "RWC_P096": "non-linear tempo changes misalign chord boundaries"}
