import pytest

from tests import assert_result
from presidio_analyzer.predefined_recognizers.iban_recognizer import IbanRecognizer


@pytest.fixture(scope="module")
def recognizer():
    return IbanRecognizer()


@pytest.fixture(scope="module")
def entities():
    return ["IBAN_CODE"]


def update_iban_checksum(iban):
    """
    Generates an IBAN, with checksum digits
    This is based on: https://www.ibantest.com/en/how-is-the-iban-check-digit-calculated
    """
    iban_no_spaces = iban.replace(" ", "")
    iban_digits = (
        (iban_no_spaces[4:] + iban_no_spaces[:2] + "00")
        .upper()
        .translate(IbanRecognizer.LETTERS)
    )
    check_digits = "{:0>2}".format(98 - (int(iban_digits) % 97))
    return iban[:2] + check_digits + iban[4:]


@pytest.mark.parametrize(
    "iban, expected_len, expected_res",
    [
        ("AL47212110090000000235698741", 1, ((0, 28),),),
        ("AL47 2121 1009 0000 0002 3569 8741", 1, ((0, 34),),),
        ("AL47 212A 1009 0000 0002 3569 8741", 0, ()),
        ("AL47 212A 1009 0000 0002 3569 874", 0, ()),
        ("AL47 2121 1009 0000 0002 3569 8740", 0, ()),
        ("AD1200012030200359100100", 1, ((0, 24),),),
        ("AD12 0001 2030 2003 5910 0100", 1, ((0, 29),),),
        ("AD12000A2030200359100100", 0, ()),
        ("AD12000A203020035910010", 0, ()),
        ("AD12 0001 2030 2003 5910 0101", 0, ()),
        ("AT611904300234573201", 1, ((0, 20),),),
        ("AT61 1904 3002 3457 3201", 1, ((0, 24),),),
        ("AT61 1904 A002 3457 3201", 0, ()),
        ("AT61 1904 3002 3457 320", 0, ()),
        ("AT61 1904 3002 3457 3202", 0, ()),
        ("AZ21NABZ00000000137010001944", 1, ((0, 28),),),
        ("AZ21 NABZ 0000 0000 1370 1000 1944", 1, ((0, 34),),),
        ("AZ21NABZ000000001370100019", 0, ()),
        ("AZ21NABZ0000000013701000194", 0, ()),
        ("AZ21NABZ00000000137010001945", 0, ()),
        ("BH67BMAG00001299123456", 1, ((0, 22),),),
        ("BH67 BMAG 0000 1299 1234 56", 1, ((0, 27),),),
        ("BH67BMA100001299123456", 0, ()),
        ("BH67BMAG0000129912345", 0, ()),
        ("BH67BMAG00001299123457", 0, ()),
        ("BY13NBRB3600900000002Z00AB00", 1, ((0, 28),),),
        ("BY13 NBRB 3600 9000 0000 2Z00 AB00", 1, ((0, 34),),),
        ("BY13NBRBA600900000002Z00AB00", 0, ()),
        ("BY13 NBRB 3600 9000 0000 2Z00 AB0", 0, ()),
        ("BY13NBRB3600900000002Z00AB01", 0, ()),
        ("BE68539007547034", 1, ((0, 16),),),
        ("BE71 0961 2345 6769", 1, ((0, 19),),),
        ("BE71 A961 2345 6769", 0, ()),
        ("BE6853900754703", 0, ()),
        ("BE71 0961 2345 6760", 0, ()),
        ("BA391290079401028494", 1, ((0, 20),),),
        ("BA39 1290 0794 0102 8494", 1, ((0, 24),),),
        ("BA39 A290 0794 0102 8494", 0, ()),
        ("BA39129007940102849", 0, ()),
        ("BA39 1290 0794 0102 8495", 0, ()),
        ("BR9700360305000010009795493P1", 1, ((0, 29),),),
        ("BR97 0036 0305 0000 1000 9795 493P 1", 1, ((0, 36),),),
        ("BR97 0036 A305 0000 1000 9795 493P 1", 0, ()),
        ("BR9700360305000010009795493P", 0, ()),
        ("BR97 0036 0305 0000 1000 9795 493P 2", 0, ()),
        ("BG80BNBG96611020345678", 1, ((0, 22),),),
        ("BG80 BNBG 9661 1020 3456 78", 1, ((0, 27),),),
        ("BG80 BNBG 9661 A020 3456 78", 0, ()),
        ("BG80BNBG9661102034567", 0, ()),
        ("BG80 BNBG 9661 1020 3456 79", 0, ()),
        ("CR05015202001026284066", 1, ((0, 22),),),
        ("CR05 0152 0200 1026 2840 66", 1, ((0, 27),),),
        ("CR05 0152 0200 1026 2840 6A", 0, ()),
        ("CR05 0152 0200 1026 2840 6", 0, ()),
        ("CR05 0152 0200 1026 2840 67", 0, ()),
        ("HR1210010051863000160", 1, ((0, 21),),),
        ("HR12 1001 0051 8630 0016 0", 1, ((0, 26),),),
        ("HR12 001 0051 8630 0016 A", 0, ()),
        ("HR121001005186300016", 0, ()),
        ("HR12 1001 0051 8630 0016 1", 0, ()),
        ("CY17002001280000001200527600", 1, ((0, 28),),),
        ("CY17 0020 0128 0000 0012 0052 7600", 1, ((0, 34),),),
        ("CY17 0020 A128 0000 0012 0052 7600", 0, ()),
        ("CY17 0020 0128 0000 0012 0052 760", 0, ()),
        ("CY17 0020 0128 0000 0012 0052 7601", 0, ()),
        ("CZ6508000000192000145399", 1, ((0, 24),),),
        ("CZ65 0800 0000 1920 0014 5399", 1, ((0, 29),),),
        ("CZ65 0800 A000 1920 0014 5399", 0, ()),
        ("CZ65 0800 0000 1920 0014 539", 0, ()),
        ("CZ65 0800 0000 1920 0014 5390", 0, ()),
        ("DK5000400440116243", 1, ((0, 18),),),
        ("DK50 0040 0440 1162 43", 1, ((0, 22),),),
        ("DK50 0040 A440 1162 43", 0, ()),
        ("DK50 0040 0440 1162 4", 0, ()),
        ("DK50 0040 0440 1162 44", 0, ()),
        ("DO28BAGR00000001212453611324", 1, ((0, 28),),),
        ("DO28 BAGR 0000 0001 2124 5361 1324", 1, ((0, 34),),),
        ("DO28 BAGR A000 0001 2124 5361 1324", 0, ()),
        ("DO28 BAGR 0000 0001 2124 5361 132", 0, ()),
        ("DO28 BAGR 0000 0001 2124 5361 1325", 0, ()),
        ("TL380080012345678910157", 1, ((0, 23),),),
        ("TL38 0080 0123 4567 8910 157", 1, ((0, 28),),),
        ("TL38 A080 0123 4567 8910 157", 0, ()),
        ("TL38 0080 0123 4567 8910 158", 0, ()),
        ("EE382200221020145685", 1, ((0, 20),),),
        ("EE38 2200 2210 2014 5685", 1, ((0, 24),),),
        ("EE38 A200 2210 2014 5685", 0, ()),
        ("EE38 2200 2210  014 5686", 0, ()),
        ("FO6264600001631634", 1, ((0, 18),),),
        ("FO62 6460 0001 6316 34", 1, ((0, 22),),),
        ("FO62 A460 0001 6316 34", 0, ()),
        ("FO62 6460 0001 6316 35", 0, ()),
        ("FI2112345600000785", 1, ((0, 18),),),
        ("FI21 1234 5600 0007 85", 1, ((0, 22),),),
        ("FI21 A234 5600 0007 85", 0, ()),
        ("FI21 1234 5600 0007 86", 0, ()),
        ("FR1420041010050500013M02606", 1, ((0, 27),),),
        ("FR14 2004 1010 0505 0001 3M02 606", 1, ((0, 33),),),
        ("FR14 A004 1010 0505 0001 3M02 606", 0, ()),
        ("FR14 2004 1010 0505 0001 3M02 607", 0, ()),
        ("GE29NB0000000101904917", 1, ((0, 22),),),
        ("GE29 NB00 0000 0101 9049 17", 1, ((0, 27),),),
        ("GE29 NBA0 0000 0101 9049 17", 0, ()),
        ("GE29 NB00 0000 0101 9049 18", 0, ()),
        ("DE89370400440532013000", 1, ((0, 22),),),
        ("DE89 3704 0044 0532 0130 00", 1, ((0, 27),),),
        ("DE89 A704 0044 0532 0130 00", 0, ()),
        ("DE89 3704 0044 0532 0130 01", 0, ()),
        ("GI75NWBK000000007099453", 1, ((0, 23),),),
        ("GI75 NWBK 0000 0000 7099 453", 1, ((0, 28),),),
        ("GI75 aWBK 0000 0000 7099 453", 0, ()),
        ("GI75 NWBK 0000 0000 7099 454", 0, ()),
        ("GR1601101250000000012300695", 1, ((0, 27),),),
        ("GR16 0110 1250 0000 0001 2300 695", 1, ((0, 33),),),
        ("GR16 A110 1250 0000 0001 2300 695", 0, ()),
        ("GR16 0110 1250 0000 0001 2300 696", 0, ()),
        ("GL8964710001000206", 1, ((0, 18),),),
        ("GL89 6471 0001 0002 06", 1, ((0, 22),),),
        ("GL89 A471 0001 0002 06", 0, ()),
        ("GL89 6471 0001 0002 07", 0, ()),
        ("GT82TRAJ01020000001210029690", 1, ((0, 28),),),
        ("GT82 TRAJ 0102 0000 0012 1002 9690", 1, ((0, 34),),),
        ("G T82 TRAJ 0102 0000 0012 1002 9690", 0, ()),
        ("GT82 TRAJ 0102 0000 0012 1002 9691", 0, ()),
        ("HU42117730161111101800000000", 1, ((0, 28),),),
        ("HU42 1177 3016 1111 1018 0000 0000", 1, ((0, 34),),),
        ("HU42 A177 3016 1111 1018 0000 0000", 0, ()),
        ("HU42 1177 3016 1111 1018 0000 0001", 0, ()),
        ("IS140159260076545510730339", 1, ((0, 26),),),
        ("IS14 0159 2600 7654 5510 7303 39", 1, ((0, 32),),),
        ("IS14 A159 2600 7654 5510 7303 39", 0, ()),
        ("IS14 0159 2600 7654 5510 7303 30", 0, ()),
        ("IE29AIBK93115212345678", 1, ((0, 22),),),
        ("IE29 AIBK 9311 5212 3456 78", 1, ((0, 27),),),
        ("IE29 AIBK A311 5212 3456 78", 0, ()),
        ("IE29 AIBK 9311 5212 3456 79", 0, ()),
        ("IL620108000000099999999", 1, ((0, 23),),),
        ("IL62 0108 0000 0009 9999 999", 1, ((0, 28),),),
        ("IL62 A108 0000 0009 9999 999", 0, ()),
        ("IL62 0108 0000 0009 9999 990", 0, ()),
        ("IT60X0542811101000000123456", 1, ((0, 27),),),
        ("IT60 X054 2811 1010 0000 0123 456", 1, ((0, 33),),),
        ("IT60 XW54 2811 1010 0000 0123 456", 0, ()),
        ("IT60 X054 2811 1010 0000 0123 457", 0, ()),
        ("JO94CBJO0010000000000131000302", 1, ((0, 30),),),
        ("JO94 CBJO 0010 0000 0000 0131 0003 02", 1, ((0, 37),),),
        ("JO94 CBJO A010 0000 0000 0131 0003 02", 0, ()),
        ("JO94 CBJO 0010 0000 0000 0131 0003 03", 0, ()),
        ("KZ86125KZT5004100100", 1, ((0, 20),),),
        ("KZ86 125K ZT50 0410 0100", 1, ((0, 24),),),
        ("KZ86 A25K ZT50 0410 0100", 0, ()),
        ("KZ86 125K ZT50 0410 0101", 0, ()),
        ("XK051212012345678906", 1, ((0, 20),),),
        ("XK05 1212 0123 4567 8906", 1, ((0, 24),),),
        ("XK05 A212 0123 4567 8906", 0, ()),
        ("XK05 1212 0123 4567 8907", 0, ()),
        ("KW81CBKU0000000000001234560101", 1, ((0, 30),),),
        ("KW81 CBKU 0000 0000 0000 1234 5601 01", 1, ((0, 37),),),
        ("KW81 aBKU 0000 0000 0000 1234 5601 01", 0, ()),
        ("KW81 CBKU 0000 0000 0000 1234 5601 02", 0, ()),
        ("LV80BANK0000435195001", 1, ((0, 21),),),
        ("LV80 BANK 0000 4351 9500 1", 1, ((0, 26),),),
        ("LV80 bANK 0000 4351 9500 1", 0, ()),
        ("LV80 BANK 0000 4351 9500 2", 0, ()),
        ("LB62099900000001001901229114", 1, ((0, 28),),),
        ("LB62 0999 0000 0001 0019 0122 9114", 1, ((0, 34),),),
        ("LB62 A999 0000 0001 0019 0122 9114", 0, ()),
        ("LB62 0999 0000 0001 0019 0122 9115", 0, ()),
        ("LI21088100002324013AA", 1, ((0, 21),),),
        ("LI21 0881 0000 2324 013A A", 1, ((0, 26),),),
        ("LI21 A881 0000 2324 013A A", 0, ()),
        ("LI21 0881 0000 2324 013A B", 0, ()),
        ("LT121000011101001000", 1, ((0, 20),),),
        ("LT12 1000 0111 0100 1000", 1, ((0, 24),),),
        ("LT12 A000 0111 0100 1000", 0, ()),
        ("LT12 1000 0111 0100 1001", 0, ()),
        ("LU280019400644750000", 1, ((0, 20),),),
        ("LU28 0019 4006 4475 0000", 1, ((0, 24),),),
        ("LU28 A019 4006 4475 0000", 0, ()),
        ("LU28 0019 4006 4475 0001", 0, ()),
        ("MT84MALT011000012345MTLCAST001S", 1, ((0, 31),),),
        ("MT84 MALT 0110 0001 2345 MTLC AST0 01S", 1, ((0, 38),),),
        ("MT84 MALT A110 0001 2345 MTLC AST0 01S", 0, ()),
        ("MT84 MALT 0110 0001 2345 MTLC AST0 01T", 0, ()),
        ("MR1300020001010000123456753", 1, ((0, 27),),),
        ("MR13 0002 0001 0100 0012 3456 753", 1, ((0, 33),),),
        ("MR13 A002 0001 0100 0012 3456 753", 0, ()),
        ("MR13 0002 0001 0100 0012 3456 754", 0, ()),
        ("MU17BOMM0101101030300200000MUR", 1, ((0, 30),),),
        ("MU17 BOMM 0101 1010 3030 0200 000M UR", 1, ((0, 37),),),
        ("MU17 BOMM A101 1010 3030 0200 000M UR", 0, ()),
        ("MU17 BOMM 0101 1010 3030 0200 000M US", 0, ()),
        ("MD24AG000225100013104168", 1, ((0, 24),),),
        ("MD24 AG00 0225 1000 1310 4168", 1, ((0, 29),),),
        ("MD24 AG00 0225 1000 1310 416", 0, ()),
        ("MD24 AG00 0225 1000 1310 4169", 0, ()),
        ("MC5811222000010123456789030", 1, ((0, 27),),),
        ("MC58 1122 2000 0101 2345 6789 030", 1, ((0, 33),),),
        ("MC58 A122 2000 0101 2345 6789 030", 0, ()),
        ("MC58 1122 2000 0101 2345 6789 031", 0, ()),
        ("ME25505000012345678951", 1, ((0, 22),),),
        ("ME25 5050 0001 2345 6789 51", 1, ((0, 27),),),
        ("ME25 A050 0001 2345 6789 51", 0, ()),
        ("ME25 5050 0001 2345 6789 52", 0, ()),
        ("NL91ABNA0417164300", 1, ((0, 18),),),
        ("NL91 ABNA 0417 1643 00", 1, ((0, 22),),),
        ("NL91 1BNA 0417 1643 00", 0, ()),
        ("NL91 ABNA 0417 1643 01", 0, ()),
        ("MK07250120000058984", 1, ((0, 19),),),
        ("MK07 2501 2000 0058 984", 1, ((0, 23),),),
        ("MK07 A501 2000 0058 984", 0, ()),
        ("MK07 2501 2000 0058 985", 0, ()),
        ("NO9386011117947", 1, ((0, 15),),),
        ("NO93 8601 1117 947", 1, ((0, 18),),),
        ("NO93 A601 1117 947", 0, ()),
        ("NO93 8601 1117 948", 0, ()),
        ("PK36SCBL0000001123456702", 1, ((0, 24),),),
        ("PK36 SCBL 0000 0011 2345 6702", 1, ((0, 29),),),
        ("PK36 SCBL A000 0011 2345 6702", 0, ()),
        ("PK36 SCBL 0000 0011 2345 6703", 0, ()),
        ("PS92PALS000000000400123456702", 1, ((0, 29),),),
        ("PS92 PALS 0000 0000 0400 1234 5670 2", 1, ((0, 36),),),
        ("PS92 PALS A000 0000 0400 1234 5670 2", 0, ()),
        ("PS92 PALS 0000 0000 0400 1234 5670 3", 0, ()),
        ("PL61109010140000071219812874", 1, ((0, 28),),),
        ("PL61 1090 1014 0000 0712 1981 2874", 1, ((0, 34),),),
        ("PL61 A090 1014 0000 0712 1981 2874", 0, ()),
        ("PL61 1090 1014 0000 0712 1981 2875", 0, ()),
        ("PT50000201231234567890154", 1, ((0, 25),),),
        ("PT50 0002 0123 1234 5678 9015 4", 1, ((0, 31),),),
        ("PT50 A002 0123 1234 5678 9015 4", 0, ()),
        ("PT50 0002 0123 1234 5678 9015 5", 0, ()),
        ("QA58DOHB00001234567890ABCDEFG", 1, ((0, 29),),),
        ("QA58 DOHB 0000 1234 5678 90AB CDEF G", 1, ((0, 36),),),
        ("QA58 0OHB 0000 1234 5678 90AB CDEF G", 0, ()),
        ("QA58 DOHB 0000 1234 5678 90AB CDEF H", 0, ()),
        ("RO49AAAA1B31007593840000", 1, ((0, 24),),),
        ("RO49 AAAA 1B31 0075 9384 0000", 1, ((0, 29),),),
        ("RO49 0AAA 1B31 0075 9384 0000", 0, ()),
        ("RO49 AAAA 1B31 0075 9384 0001", 0, ()),
        ("SM86U0322509800000000270100", 1, ((0, 27),),),
        ("SM86 U032 2509 8000 0000 0270 100", 1, ((0, 33),),),
        ("SM86 0032 2509 8000 0000 0270 100", 0, ()),
        ("SM86 U032 2509 8000 0000 0270 101", 0, ()),
        ("SA0380000000608010167519", 1, ((0, 24),),),
        ("SA03 8000 0000 6080 1016 7519", 1, ((0, 29),),),
        ("SA03 A000 0000 6080 1016 7519", 0, ()),
        ("SA03 8000 0000 6080 1016 7510", 0, ()),
        ("RS35260005601001611379", 1, ((0, 22),),),
        ("RS35 2600 0560 1001 6113 79", 1, ((0, 27),),),
        ("RS35 A600 0560 1001 6113 79", 0, ()),
        ("RS35 2600 0560 1001 6113 70", 0, ()),
        ("SK3112000000198742637541", 1, ((0, 24),),),
        ("SK31 1200 0000 1987 4263 7541", 1, ((0, 29),),),
        ("SK31 A200 0000 1987 4263 7541", 0, ()),
        ("SK31 1200 0000 1987 4263 7542", 0, ()),
        ("SI56263300012039086", 1, ((0, 19),),),
        ("SI56 2633 0001 2039 086", 1, ((0, 23),),),
        ("SI56 A633 0001 2039 086", 0, ()),
        ("SI56 2633 0001 2039 087", 0, ()),
        ("ES9121000418450200051332", 1, ((0, 24),),),
        ("ES91 2100 0418 4502 0005 1332", 1, ((0, 29),),),
        ("ES91 A100 0418 4502 0005 1332", 0, ()),
        ("ES91 2100 0418 4502 0005 1333", 0, ()),
        ("SE4550000000058398257466", 1, ((0, 24),),),
        ("SE45 5000 0000 0583 9825 7466", 1, ((0, 29),),),
        ("SE45 A000 0000 0583 9825 7466", 0, ()),
        ("SE45 5000 0000 0583 9825 7467", 0, ()),
        ("CH9300762011623852957", 1, ((0, 21),),),
        ("CH93 0076 2011 6238 5295 7", 1, ((0, 26),),),
        ("CH93 A076 2011 6238 5295 7", 0, ()),
        ("CH93 0076 2011 6238 5295 8", 0, ()),
        ("TN5910006035183598478831", 1, ((0, 24),),),
        ("TN59 1000 6035 1835 9847 8831", 1, ((0, 29),),),
        ("TN59 A000 6035 1835 9847 8831", 0, ()),
        ("CH93 0076 2011 6238 5295 9", 0, ()),
        ("TR330006100519786457841326", 1, ((0, 26),),),
        ("TR33 0006 1005 1978 6457 8413 26", 1, ((0, 32),),),
        ("TR33 A006 1005 1978 6457 8413 26", 0, ()),
        ("TR33 0006 1005 1978 6457 8413 27", 0, ()),
        ("AE070331234567890123456", 1, ((0, 23),),),
        ("AE07 0331 2345 6789 0123 456", 1, ((0, 28),),),
        ("AE07 A331 2345 6789 0123 456", 0, ()),
        ("AE07 0331 2345 6789 0123 457", 0, ()),
        ("GB29NWBK60161331926819", 1, ((0, 22),),),
        ("GB29 NWBK 6016 1331 9268 19", 1, ((0, 27),),),
        ("GB29 1WBK 6016 1331 9268 19", 0, ()),
        ("GB29 NWBK 6016 1331 9268 10", 0, ()),
        ("VA59001123000012345678", 1, ((0, 22),),),
        ("VA59 0011 2300 0012 3456 78", 1, ((0, 27),),),
        ("VA59 A011 2300 0012 3456 78", 0, ()),
        ("VA59 0011 2300 0012 3456 79", 0, ()),
        ("VG96VPVG0000012345678901", 1, ((0, 24),),),
        ("VG96 VPVG 0000 0123 4567 8901", 1, ((0, 29),),),
        ("VG96 VPVG A000 0123 4567 8901", 0, ()),
        ("VG96 VPVG 0000 0123 4567 8902", 0, ()),
        (
            "this is an iban VG96 VPVG 0000 0123 4567 8901 in a sentence",
            1,
            ((16, 45),),
        ),
        (
            "this is an iban VG96 VPVG 0000 0123 4567 8901 X in a sentence",
            1,
            ((16, 45),),
        ),
        ("AB150120690000003111141", 0, ()),
        ("AB150120690000003111141", 0, ()),
        ("IL15 0120 6900 0000", 0, ()),
        ("IL15 0120 6900 0000 3111 0120 6900 0000 3111 141", 0, ()),
        ("IL150120690000003111141", 0, ()),
        ("AM47212110090000000235698740", 0, ()),
        (
            "list of ibans: AL47212110090000000235698741, AL47212110090000000235698741",
            2,
            ((15, 43), (45, 73),),
        ),
        (
            "Dash as iban separator: AL47-2121-1009-0000-0002-3569-8741",
            1,
            ((24, 58),),
        ),
        (
            "Slash as iban separator: AL47/2121/1009/0000/0002/3569/8741",
            1,
            ((25, 59),),
        ),
    ],
)
def test_all_ibans(iban, expected_len, expected_res, recognizer, entities, max_score):
    results = recognizer.analyze(iban, entities)
    assert len(results) == expected_len
    for res, (start, end) in zip(results, expected_res):
        assert_result(res, entities[0], start, end, max_score)
