# Experiment 2C3 Final Report

## Opening summary

Classification: **MATCHED MULTI-DESTINATION FEEDBACK MATURES AND SCALES**

Frozen rule: C1 regressions and integrity pass; C3 or C4 passes the frozen 100M gain, growth, share, recovery, gap, and 18/20 paired-win thresholds.

100M real losses: C1=5.5957053900, C2=5.5999835491, C3=6.4834810495, C4=6.8263012886.

100M shuffled-real gaps: C1=0.1415745974, C2=0.1397536993, C3=0.1799920797, C4=0.1662387133.

Matched gains (25M → 100M): C2=0.0033987045→0.0142239332, C3=0.0126734018→0.0743097067, C4=0.0154254198→0.0690653801.

100M recovery fractions: C1=0.199454, C2=0.197525, C3=0.148793, C4=0.124571.

B1 remained dominant under the frozen 0.020 matched-gain criterion: **NO**. However, the B1-only pathway still accounts for about 82.3% of total recovery in both C3 and C4, so B1 remains the primary—not exclusive—gateway.

Later readers with positive individual alignment value: C2/B2, C3/B2, C3/B3, C4/B2, C4/B3, C4/B4.

C3/C4 B1 v17 routing at 100M: 0.266400 / 0.321104. C3 is no longer v17-dominant (v16=0.289058), while C4 remains v17-dominant (v16=0.289413).

Multi-reader self recurrence triggered: C3, C4. Neither transferred positive overall recovery zero-shot: C3=-0.0200496689 and C4=-0.1093781534, although their additional readers retained positive self matched gains of 0.0256703898 and 0.0534682458.

## Provenance

- 2C2 frozen tag: `experiment-2c2-cumulative-low-kv-final`
- 2C2 parent commit: `5853308bc172150b05bafb32222f2461230adac5`
- 2C3 branch: `experiment-2c3-cumulative-reader-scaling-100m`
- Implementation commit: `792dc701f29b449c611b0a524ef4277a5f982403`
- Results commit: `fc0b4acda5d252cdbe7fc5d3781edffba7520586`
- Final-report commit: the immutable commit containing this file
- Base checkpoint SHA256: `6e3a6dbd9fe3d81d580c1667caae7779e926d464ce3f6d962a8591ceeceefa91`

## Final main table

| Config | Masked | 25M Real | 100M Real | 100M Shuffled | 100M Generic | 100M Gap | Recovery % | B1-only | Matched gain | Matched share | Real wins |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 5.9736744881 | 5.8353391409 | 5.5957053900 | 5.7372799873 | 5.6806090117 | 0.1415745974 | 19.945388 | 5.5957053900 | n/a | n/a | 20/20 |
| C2 | 5.9744511843 | 5.8394160986 | 5.5999835491 | 5.7397372484 | 5.7058311462 | 0.1397536993 | 19.752520 | 5.6142074823 | 0.0142239332 | 0.037984 | 20/20 |
| C3 | 6.9038509846 | 6.7783385992 | 6.4834810495 | 6.6634731293 | 6.7402603149 | 0.1799920797 | 14.879316 | 6.5577907562 | 0.0743097067 | 0.176772 | 20/20 |
| C4 | 7.2172823668 | 7.1104239941 | 6.8263012886 | 6.9925400019 | 7.0602352142 | 0.1662387133 | 12.457070 | 6.8953666687 | 0.0690653801 | 0.176646 | 20/20 |

## Maturation

| Config | Metric | 25M | 50M | 75M | 100M |
|---|---|---:|---:|---:|---:|
| C1 | Real loss | 5.8353391409 | 5.7143192530 | 5.6375406504 | 5.5957053900 |
| C1 | Specific gap | 0.0412521124 | 0.0810096741 | 0.1136229038 | 0.1415745974 |
| C1 | Recovery fraction | 0.0729994097 | 0.1368614707 | 0.1773774544 | 0.1994538750 |
| C1 | Matched gain | n/a | n/a | n/a | n/a |
| C1 | Matched share | n/a | n/a | n/a | n/a |
| C2 | Real loss | 5.8394160986 | 5.7190018415 | 5.6417348385 | 5.5999835491 |
| C2 | Specific gap | 0.0400157690 | 0.0778317928 | 0.1107973576 | 0.1397536993 |
| C2 | Recovery fraction | 0.0712286715 | 0.1347451088 | 0.1755021161 | 0.1975251989 |
| C2 | Matched gain | 0.0033987045 | 0.0074138880 | 0.0110129356 | 0.0142239332 |
| C2 | Matched share | 0.0251690478 | 0.0290229284 | 0.0331000739 | 0.0379844128 |
| C3 | Real loss | 6.7783385992 | 6.6493239164 | 6.5495995283 | 6.4834810495 |
| C3 | Specific gap | 0.0583660364 | 0.1023851156 | 0.1424623728 | 0.1799920797 |
| C3 | Recovery fraction | 0.0444260716 | 0.0900918082 | 0.1253900204 | 0.1487931631 |
| C3 | Matched gain | 0.0126734018 | 0.0307739973 | 0.0513857126 | 0.0743097067 |
| C3 | Matched share | 0.1009733167 | 0.1209065799 | 0.1450543441 | 0.1767721725 |
| C4 | Real loss | 7.1104239941 | 6.9883648157 | 6.8933439255 | 6.8263012886 |
| C4 | Specific gap | 0.0556101084 | 0.0942162752 | 0.1324033022 | 0.1662387133 |
| C4 | Recovery fraction | 0.0340462056 | 0.0729355483 | 0.1032102070 | 0.1245706988 |
| C4 | Matched gain | 0.0154254198 | 0.0334849358 | 0.0511325121 | 0.0690653801 |
| C4 | Matched share | 0.1443538716 | 0.1462750917 | 0.1578463855 | 0.1766463493 |

## M100 matched-gain pairs

| Config | B1-only | All-real | Matched gain | All-real wins | B1-only wins |
|---|---:|---:|---:|---:|---:|
| C2 | 5.6142074823 | 5.5999835491 | 0.0142239332 | 20/20 | 0/20 |
| C3 | 6.5577907562 | 6.4834810495 | 0.0743097067 | 20/20 | 0/20 |
| C4 | 6.8953666687 | 6.8263012886 | 0.0690653801 | 20/20 | 0/20 |

## Progressive activation

| Config | Active readers | Loss | Incremental gain |
|---|---|---:|---:|
| C2 | none | 5.9744511843 | 0.0000000000 |
| C2 | B1 | 5.6142074823 | 0.3602437019 |
| C2 | B1+B2 | 5.5999835491 | 0.0142239332 |
| C3 | none | 6.9038509846 | 0.0000000000 |
| C3 | B1 | 6.5577907562 | 0.3460602283 |
| C3 | B1+B2 | 6.5462275743 | 0.0115631819 |
| C3 | B1+B2+B3 | 6.4834810495 | 0.0627465248 |
| C4 | none | 7.2172823668 | 0.0000000000 |
| C4 | B1 | 6.8953666687 | 0.3219156981 |
| C4 | B1+B2 | 6.8875682831 | 0.0077983856 |
| C4 | B1+B2+B3 | 6.8526740313 | 0.0348942518 |
| C4 | B1+B2+B3+B4 | 6.8263012886 | 0.0263727427 |

## Leave-one-reader-out

| Config | Reader removed | Loss | Delta vs all-real | Positive batches |
|---|---|---:|---:|---:|
| C1 | B1 | 5.9736744881 | 0.3779690981 | 20/20 |
| C2 | B1 | 5.9553160906 | 0.3553325415 | 20/20 |
| C2 | B2 | 5.6142074823 | 0.0142239332 | 20/20 |
| C3 | B1 | 6.8282583714 | 0.3447773218 | 20/20 |
| C3 | B2 | 6.4940000057 | 0.0105189562 | 20/20 |
| C3 | B3 | 6.5462275743 | 0.0627465248 | 20/20 |
| C4 | B1 | 7.1473783970 | 0.3210771084 | 20/20 |
| C4 | B2 | 6.8336025238 | 0.0073012352 | 20/20 |
| C4 | B3 | 6.8611626625 | 0.0348613739 | 20/20 |
| C4 | B4 | 6.8526740313 | 0.0263727427 | 20/20 |

## Per-reader sequence alignment

| Config | Reader shuffled | Loss | Delta vs all-real | Positive batches |
|---|---|---:|---:|---:|
| C1 | B1 | 5.7372799873 | 0.1415745974 | 20/20 |
| C2 | B1 | 5.7241908550 | 0.1242073059 | 20/20 |
| C2 | B2 | 5.6068778038 | 0.0068942547 | 20/20 |
| C3 | B1 | 6.6305847406 | 0.1471036911 | 20/20 |
| C3 | B2 | 6.4872222185 | 0.0037411690 | 20/20 |
| C3 | B3 | 6.4996248007 | 0.0161437511 | 20/20 |
| C4 | B1 | 6.9687354803 | 0.1424341917 | 20/20 |
| C4 | B2 | 6.8297911644 | 0.0034898758 | 19/20 |
| C4 | B3 | 6.8360137224 | 0.0097124338 | 20/20 |
| C4 | B4 | 6.8378438473 | 0.0115425587 | 20/20 |

## Final reader routing

| Config | Destination | Gate | Query norm | RMS displacement | Entropy | v16 | v17 | v20 | v24 | Feedback RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | B1 | 0.1138940752 | 1.5761964321 | 1.5883628130 | 0.5843298912 | 0.4224672645 | 0.1342636142 | 0.2162808836 | 0.2269882388 | 0.0212922591 |
| C2 | B1 | 0.1135615781 | 1.5866315365 | 1.5907974243 | 0.5976912975 | 0.4064473808 | 0.1514856447 | 0.2153563142 | 0.2267106593 | 0.0211785804 |
| C2 | B2 | -0.1202163324 | 1.7180764675 | 1.6370638609 | 0.4387483612 | 0.4214822978 | 0.0633798892 | 0.2858556047 | 0.2292822070 | 0.0240668831 |
| C3 | B1 | 0.1132921651 | 1.4169582129 | 1.3516972065 | 0.7049956709 | 0.2890575960 | 0.2663996942 | 0.2497051947 | 0.1948375106 | 0.0209327744 |
| C3 | B2 | -0.1204259098 | 1.7494565248 | 1.6375690699 | 0.4424528152 | 0.3656778991 | 0.0219498427 | 0.3652179539 | 0.2471543059 | 0.0236497059 |
| C3 | B3 | -0.1218926460 | 1.9475795031 | 1.8956040144 | 0.2678689837 | 0.2642820172 | 0.0050387548 | 0.5006122604 | 0.2300669655 | 0.0268689330 |
| C4 | B1 | 0.1138466299 | 1.3588900566 | 1.2913978100 | 0.7253497809 | 0.2894133255 | 0.3211035550 | 0.2301270224 | 0.1593561016 | 0.0210460218 |
| C4 | B2 | -0.1186070889 | 1.3569656610 | 1.2502394915 | 0.4556842700 | 0.4648233086 | 0.0170815250 | 0.2448481143 | 0.2732470520 | 0.0237632947 |
| C4 | B3 | -0.1201642081 | 1.3823721409 | 1.2951290607 | 0.3235465854 | 0.3672446504 | 0.0097638314 | 0.3774429858 | 0.2455485336 | 0.0245555330 |
| C4 | B4 | -0.1174840480 | 1.1289917231 | 1.0507411957 | 0.4121189415 | 0.5782022417 | 0.0985606406 | 0.1614049137 | 0.1618321978 | 0.0244103241 |

## B1 evolution

| Config | Milestone | Gate | Query norm | v16 | v17 | v20 | v24 | Entropy | Feedback RMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | 100M | 0.1138940752 | 1.5761964321 | 0.4224672645 | 0.1342636142 | 0.2162808836 | 0.2269882388 | 0.5843298912 | 0.0212922591 |
| C1 | 25M | 0.0287788194 | 0.6052711010 | 0.4061413527 | 0.1650304757 | 0.1690091498 | 0.2598190166 | 0.7376077384 | 0.0051791935 |
| C1 | 50M | 0.0575050488 | 1.0308834314 | 0.4201693878 | 0.1458385758 | 0.2046197884 | 0.2293722406 | 0.6137965679 | 0.0107665394 |
| C1 | 75M | 0.0861299857 | 1.3569902182 | 0.4223711729 | 0.1358548649 | 0.2135175310 | 0.2282564282 | 0.5821851522 | 0.0161945770 |
| C2 | 100M | 0.1135615781 | 1.5866315365 | 0.4064473808 | 0.1514856447 | 0.2153563142 | 0.2267106593 | 0.5976912975 | 0.0211785804 |
| C2 | 25M | 0.0287782904 | 0.5994858146 | 0.3858857229 | 0.1913199544 | 0.1599657089 | 0.2628286131 | 0.7650935531 | 0.0051580094 |
| C2 | 50M | 0.0575026087 | 1.0253145695 | 0.4004385814 | 0.1682439946 | 0.1992731087 | 0.2320443086 | 0.6435298115 | 0.0107046994 |
| C2 | 75M | 0.0861212313 | 1.3649624586 | 0.4072277486 | 0.1529775508 | 0.2107563317 | 0.2290383682 | 0.6039131254 | 0.0161087855 |
| C3 | 100M | 0.1132921651 | 1.4169582129 | 0.2890575960 | 0.2663996942 | 0.2497051947 | 0.1948375106 | 0.7049956709 | 0.0209327744 |
| C3 | 25M | 0.0287692901 | 0.4221085906 | 0.1100261644 | 0.6072065294 | 0.0987665549 | 0.1840007469 | 0.7865854532 | 0.0055082500 |
| C3 | 50M | 0.0574586652 | 0.8060903549 | 0.2285470076 | 0.3699186444 | 0.1828425489 | 0.2186918028 | 0.7552516192 | 0.0108013325 |
| C3 | 75M | 0.0859539583 | 1.1579928398 | 0.2836530477 | 0.2836883731 | 0.2316584356 | 0.2010001406 | 0.7347983211 | 0.0158500647 |
| C4 | 100M | 0.1138466299 | 1.3588900566 | 0.2894133255 | 0.3211035550 | 0.2301270224 | 0.1593561016 | 0.7253497809 | 0.0210460218 |
| C4 | 25M | 0.0287670735 | 0.4329772592 | 0.0265033858 | 0.8647929877 | 0.0400583327 | 0.0686452916 | 0.3853874221 | 0.0063101114 |
| C4 | 50M | 0.0574743636 | 0.7717044353 | 0.1961469211 | 0.4542429477 | 0.1603986450 | 0.1892114870 | 0.7308861554 | 0.0110562424 |
| C4 | 75M | 0.0860628560 | 1.1042594910 | 0.2614216901 | 0.3594130635 | 0.2129136764 | 0.1662515759 | 0.7534268051 | 0.0159855179 |

## B1 query cosine matrices

### 100M

| | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| C1 | 1.0000000000 | 0.9794611335 | 0.8337048888 | 0.7648984194 |
| C2 | 0.9794611335 | 1.0000000000 | 0.8852924109 | 0.8123152852 |
| C3 | 0.8337048888 | 0.8852924109 | 1.0000001192 | 0.9350919724 |
| C4 | 0.7648984194 | 0.8123152852 | 0.9350919724 | 0.9999998808 |

### 25M

| | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| C1 | 0.9999998808 | 0.9869624376 | 0.5890696645 | 0.4826385677 |
| C2 | 0.9869624376 | 1.0000000000 | 0.6344634295 | 0.5265815258 |
| C3 | 0.5890696645 | 0.6344634295 | 0.9999997616 | 0.9471350908 |
| C4 | 0.4826385677 | 0.5265815258 | 0.9471350908 | 1.0000000000 |

### 50M

| | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| C1 | 0.9999998212 | 0.9852263927 | 0.7864971161 | 0.7065239549 |
| C2 | 0.9852263927 | 0.9999998808 | 0.8214941621 | 0.7376314998 |
| C3 | 0.7864971161 | 0.8214941621 | 1.0000000000 | 0.9473963976 |
| C4 | 0.7065239549 | 0.7376314998 | 0.9473963976 | 1.0000001192 |

### 75M

| | C1 | C2 | C3 | C4 |
|---|---:|---:|---:|---:|
| C1 | 1.0000001192 | 0.9816612601 | 0.8270187974 | 0.7598125935 |
| C2 | 0.9816612601 | 1.0000000000 | 0.8730900288 | 0.8018454909 |
| C3 | 0.8270187974 | 0.8730900288 | 1.0000000000 | 0.9412944913 |
| C4 | 0.7598125935 | 0.8018454909 | 0.9412944913 | 1.0000000000 |

## Generic comparison

| Config | Masked | Generic | Shuffled | Real | Generic-real | Shuffled-real |
|---|---:|---:|---:|---:|---:|---:|
| C1 | 5.9736744881 | 5.6806090117 | 5.7372799873 | 5.5957053900 | 0.0849036217 | 0.1415745974 |
| C2 | 5.9744511843 | 5.7058311462 | 5.7397372484 | 5.5999835491 | 0.1058475971 | 0.1397536993 |
| C3 | 6.9038509846 | 6.7402603149 | 6.6634731293 | 6.4834810495 | 0.2567792654 | 0.1799920797 |
| C4 | 7.2172823668 | 7.0602352142 | 6.9925400019 | 6.8263012886 | 0.2339339256 | 0.1662387133 |

## Conditional self recurrence

| Config | Status | Teacher real | Teacher shuffled | Teacher gap | Teacher recovery | Self real | Self shuffled | Self gap | Self recovery | Self/teacher | Self B1-only | Self all-real | Self matched gain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 | TRIGGERED | 5.5957053900 | 5.7372799873 | 0.1415745974 | 0.3779690981 | 5.6952485141 | 5.7253828784 | 0.0301343642 | 0.2784259739 | 0.736637 | 5.6952485141 | 5.6952485141 | 0.0000000000 |
| C2 | SELF TEST NOT TRIGGERED | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| C3 | TRIGGERED | 6.4834810495 | 6.6634731293 | 0.1799920797 | 0.4203699350 | 6.9239006535 | 6.8975200741 | -0.0263805794 | -0.0200496689 | -0.047695 | 6.9495710433 | 6.9239006535 | 0.0256703898 |
| C4 | TRIGGERED | 6.8263012886 | 6.9925400019 | 0.1662387133 | 0.3909810781 | 7.3266605202 | 7.2942764722 | -0.0323840480 | -0.1093781534 | -0.279753 | 7.3801287660 | 7.3266605202 | 0.0534682458 |

## Performance

Times are wall-clock minutes. VRAM values are MiB. The four configurations trained and evaluated concurrently on four independent A100-SXM4-80GB GPUs.

| Config/GPU | 25→50M train | 50→75M train | 75→100M train | Total update time | Teacher eval | Self eval | Targets/s | Peak allocated | Peak reserved |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| C1 / GPU 0 | 25.624 | 26.005 | 25.088 | 76.718 | 4.792 | 79.426 | 16319.8 | 56917.1 | 70410.0 |
| C2 / GPU 1 | 25.629 | 25.612 | 25.097 | 76.339 | 6.580 | 0.000 | 16369.0 | 57778.7 | 71276.0 |
| C3 / GPU 2 | 25.613 | 25.644 | 25.160 | 76.418 | 7.667 | 102.480 | 16352.4 | 58639.3 | 72140.0 |
| C4 / GPU 3 | 25.672 | 25.654 | 25.106 | 76.432 | 8.774 | 102.977 | 16349.2 | 59498.2 | 73006.0 |

Total four-GPU elapsed wall time recorded by the protocol: **188.183 minutes (3:08:11)**.

## Scientific questions

- Q1: **Not by the frozen materiality threshold.** C2/B2 improved from 0.0033987045 to 0.0142239332 matched gain (+0.0108252287; 4.19×), but remained below 0.020.
- Q2: **Yes.** C3's B2+B3 matched contribution reached 0.0743097067, growing by 0.0616363049 from 25M and winning 20/20 paired batches.
- Q3: **Yes.** C4's B2+B3+B4 matched contribution reached 0.0690653801, growing by 0.0536399603 from 25M and winning 20/20 paired batches.
- Q4: **Yes, as a majority share.** C3/C4 matched shares are 0.176772/0.176646, leaving about 82.3% of recovery attributable to the B1-only pathway under this intervention; the later destinations are nevertheless material.
- Q5: **Yes.** C3/C4 specific gaps grew by 0.1216260433/0.1106286049 to 0.1799920797/0.1662387133.
- Q6: **Yes.** Real aligned state beats the generic template by 0.0849036217, 0.1058475971, 0.2567792654, and 0.2339339256 for C1–C4.
- Q7: **Every later reader showed positive alignment value.** C2/B2=0.0068942547 (20/20); C3/B2=0.0037411690 (20/20), C3/B3=0.0161437511 (20/20); C4/B2=0.0034898758 (19/20), C4/B3=0.0097124338 (20/20), C4/B4=0.0115425587 (20/20).
- Q8: **Only for C4.** C3/B1 shifted to v16=0.289058 over v17=0.266400; C4/B1 remained v17-dominant at 0.321104 versus v16=0.289413.
- Q9: **Maturation helps but does not remove depth degradation.** Recovery fractions are 0.199454, 0.197525, 0.148793, and 0.124571 for C1–C4, still decreasing at deeper cumulative masks.
- Q10: **No successful multi-reader teacher configuration transferred positive overall recovery zero-shot.** C3/C4 triggered but produced self recoveries of -0.0200496689/-0.1093781534. Only C1 transferred positive recovery (0.2784259739; self/teacher ratio 0.736637). C3/C4's positive self matched gains show that later readers help relative to their self B1-only baselines, but do not close the overall recurrence loop.

## Next-experiment decisions

- A: No. Insufficient training was a material explanation for the weak 25M C3/C4 contributions because both matured strongly by 100M; it is not an adequate explanation for C2/B2 remaining below 0.020 after 100M.
- B: No multi-reader configuration established positive zero-shot self-recurrent recovery. C3/C4 are strong teacher-assisted readers but are not yet validated candidates for self-reader adaptation without a separate preregistered intervention.
- C: Not for immediate implementation. Later-reader alignment is clear, but negative C3/C4 self recovery means any writer experiment must be separately preregistered and justified as a loop-closing intervention.
- D: If writers are later authorized, alternate frozen reader/writer phases; do not co-train in 2C3.
- E: No iterative loop is authorized; any such test requires separate preregistration.
- F: Focus primarily on B1 while retaining selective later destinations: B1 still supplies about 82.3% of C3/C4 recovery, but equivalent-memory B2/B3/B4 readers now make material contributions.
- G: The combination of strong teacher-assisted later readers and failed zero-shot multi-reader recurrence is consistent with later layers needing transformed outputs of earlier recurrent computation rather than independent copies of the same bank. This is a motivated hypothesis, not established by 2C3.

## Integrity and stopping

All frozen audit checks: **PASS**.

| Audit check | Result |
|---|---|
| 2c2_final_checkpoint_shas_exact | PASS |
| 2c2_frozen_commit_exact | PASS |
| all_adam_moments_restored | PASS |
| all_adam_steps_restored | PASS |
| all_gradients_finite | PASS |
| all_later_blocks_retain_kv | PASS |
| all_loader_states_restored | PASS |
| all_losses_finite | PASS |
| all_reader_gradients_nonzero | PASS |
| all_rng_states_restored | PASS |
| base_checkpoint_sha_exact | PASS |
| base_frozen | PASS |
| c1_100m_historical_regression | PASS |
| c1_50m_historical_regression | PASS |
| canonical_validation_hash_exact | PASS |
| exactly_100139008_total_reader_targets_per_config | PASS |
| exactly_143_new_2c3_updates_per_config | PASS |
| exactly_191_total_reader_updates_per_config | PASS |
| exactly_74973184_new_2c3_targets_per_config | PASS |
| final_checkpoints_strict_reload | PASS |
| forced_m75_fresh_process_restart | PASS |
| fresh_process_2c2_to_2c3_resume | PASS |
| future_causality | PASS |
| hellaswag_not_run | PASS |
| identical_c1_c4_batch_streams | PASS |
| m75_checkpoint_strict_reload | PASS |
| no_additional_mask_depths | PASS |
| no_auxiliary_objective | PASS |
| no_bptt | PASS |
| no_iterative_loops | PASS |
| only_intended_blocks_masked | PASS |
| reader_destination_mapping_exact | PASS |
| row_isolation | PASS |
| self_evaluation_zero_optimizer | PASS |
| single_implementation_commit | PASS |
| source_checkpoints_update_48_exact | PASS |
| teacher_frozen | PASS |
| trainable_parameter_counts_exact | PASS |
| writers_never_active | PASS |

- Starting updates/config: 48
- New 2C3 updates/config: 143
- Final updates/config: 191
- New optimizer updates across four configurations: 572
- New training targets across four configurations: 299892736
- No writers, self training, BPTT, iterative loops, extra masks, HellaSwag, or follow-on optimization ran.

# EXPERIMENT 2C3 COMPLETE