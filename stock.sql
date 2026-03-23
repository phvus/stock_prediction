--
-- PostgreSQL database dump
--

\restrict wKqHBNYrdkeaOFcuhjBlO5MJsdl2RvwotuSk1WuCUtFw7zru0wMzyeZA18mZ1mV

-- Dumped from database version 18.3
-- Dumped by pg_dump version 18.3

SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET transaction_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SELECT pg_catalog.set_config('search_path', '', false);
SET check_function_bodies = false;
SET xmloption = content;
SET client_min_messages = warning;
SET row_security = off;

SET default_tablespace = '';

SET default_table_access_method = heap;

--
-- Name: acb; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.acb (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.acb OWNER TO postgres;

--
-- Name: anv; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.anv (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.anv OWNER TO postgres;

--
-- Name: bcm; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.bcm (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.bcm OWNER TO postgres;

--
-- Name: bid; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.bid (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.bid OWNER TO postgres;

--
-- Name: bmp; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.bmp (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.bmp OWNER TO postgres;

--
-- Name: bsi; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.bsi (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.bsi OWNER TO postgres;

--
-- Name: bsr; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.bsr (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.bsr OWNER TO postgres;

--
-- Name: bvh; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.bvh (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.bvh OWNER TO postgres;

--
-- Name: bwe; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.bwe (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.bwe OWNER TO postgres;

--
-- Name: cii; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cii (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.cii OWNER TO postgres;

--
-- Name: cmg; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cmg (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.cmg OWNER TO postgres;

--
-- Name: company_info; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.company_info (
    symbol character varying(255) NOT NULL,
    icb_name2 character varying(255),
    listing_date date,
    ceo_name character varying(255)
);


ALTER TABLE public.company_info OWNER TO postgres;

--
-- Name: ctd; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ctd (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.ctd OWNER TO postgres;

--
-- Name: ctg; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ctg (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.ctg OWNER TO postgres;

--
-- Name: ctr; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ctr (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.ctr OWNER TO postgres;

--
-- Name: cts; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.cts (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.cts OWNER TO postgres;

--
-- Name: dbc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dbc (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.dbc OWNER TO postgres;

--
-- Name: dcm; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dcm (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.dcm OWNER TO postgres;

--
-- Name: dgc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dgc (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.dgc OWNER TO postgres;

--
-- Name: dgw; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dgw (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.dgw OWNER TO postgres;

--
-- Name: dig; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dig (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.dig OWNER TO postgres;

--
-- Name: dpm; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dpm (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.dpm OWNER TO postgres;

--
-- Name: dse; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dse (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.dse OWNER TO postgres;

--
-- Name: dxg; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dxg (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.dxg OWNER TO postgres;

--
-- Name: dxs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.dxs (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.dxs OWNER TO postgres;

--
-- Name: eib; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.eib (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.eib OWNER TO postgres;

--
-- Name: evf; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.evf (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.evf OWNER TO postgres;

--
-- Name: fpt; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.fpt (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.fpt OWNER TO postgres;

--
-- Name: frt; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.frt (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.frt OWNER TO postgres;

--
-- Name: fts; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.fts (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.fts OWNER TO postgres;

--
-- Name: gas; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.gas (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.gas OWNER TO postgres;

--
-- Name: gee; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.gee (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.gee OWNER TO postgres;

--
-- Name: gex; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.gex (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.gex OWNER TO postgres;

--
-- Name: gmd; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.gmd (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.gmd OWNER TO postgres;

--
-- Name: gvr; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.gvr (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.gvr OWNER TO postgres;

--
-- Name: hag; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hag (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.hag OWNER TO postgres;

--
-- Name: hcm; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hcm (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.hcm OWNER TO postgres;

--
-- Name: hdb; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hdb (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.hdb OWNER TO postgres;

--
-- Name: hdc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hdc (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.hdc OWNER TO postgres;

--
-- Name: hdg; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hdg (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.hdg OWNER TO postgres;

--
-- Name: hhv; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hhv (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.hhv OWNER TO postgres;

--
-- Name: hpg; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hpg (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.hpg OWNER TO postgres;

--
-- Name: hsg; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.hsg (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.hsg OWNER TO postgres;

--
-- Name: ht1; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ht1 (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.ht1 OWNER TO postgres;

--
-- Name: imp; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.imp (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.imp OWNER TO postgres;

--
-- Name: kbc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.kbc (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.kbc OWNER TO postgres;

--
-- Name: kdc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.kdc (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.kdc OWNER TO postgres;

--
-- Name: kdh; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.kdh (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.kdh OWNER TO postgres;

--
-- Name: kos; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.kos (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.kos OWNER TO postgres;

--
-- Name: lpb; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.lpb (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.lpb OWNER TO postgres;

--
-- Name: mbb; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.mbb (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.mbb OWNER TO postgres;

--
-- Name: msb; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.msb (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.msb OWNER TO postgres;

--
-- Name: msn; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.msn (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.msn OWNER TO postgres;

--
-- Name: mwg; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.mwg (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.mwg OWNER TO postgres;

--
-- Name: nab; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nab (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.nab OWNER TO postgres;

--
-- Name: nkg; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nkg (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.nkg OWNER TO postgres;

--
-- Name: nlg; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nlg (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.nlg OWNER TO postgres;

--
-- Name: nt2; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nt2 (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.nt2 OWNER TO postgres;

--
-- Name: nvl; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.nvl (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.nvl OWNER TO postgres;

--
-- Name: ocb; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ocb (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.ocb OWNER TO postgres;

--
-- Name: pan; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.pan (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.pan OWNER TO postgres;

--
-- Name: pc1; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.pc1 (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.pc1 OWNER TO postgres;

--
-- Name: pdr; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.pdr (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.pdr OWNER TO postgres;

--
-- Name: phr; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.phr (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.phr OWNER TO postgres;

--
-- Name: plx; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.plx (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.plx OWNER TO postgres;

--
-- Name: pnj; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.pnj (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.pnj OWNER TO postgres;

--
-- Name: pow; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.pow (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.pow OWNER TO postgres;

--
-- Name: pvd; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.pvd (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.pvd OWNER TO postgres;

--
-- Name: pvt; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.pvt (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.pvt OWNER TO postgres;

--
-- Name: ree; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ree (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.ree OWNER TO postgres;

--
-- Name: sab; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sab (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.sab OWNER TO postgres;

--
-- Name: sbt; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sbt (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.sbt OWNER TO postgres;

--
-- Name: scs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.scs (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.scs OWNER TO postgres;

--
-- Name: shb; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.shb (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.shb OWNER TO postgres;

--
-- Name: sip; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sip (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.sip OWNER TO postgres;

--
-- Name: sjs; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.sjs (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.sjs OWNER TO postgres;

--
-- Name: ssb; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ssb (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.ssb OWNER TO postgres;

--
-- Name: ssi; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.ssi (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.ssi OWNER TO postgres;

--
-- Name: stb; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.stb (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.stb OWNER TO postgres;

--
-- Name: szc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.szc (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.szc OWNER TO postgres;

--
-- Name: tch; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.tch (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.tch OWNER TO postgres;

--
-- Name: tpb; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.tpb (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.tpb OWNER TO postgres;

--
-- Name: vcb; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vcb (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vcb OWNER TO postgres;

--
-- Name: vcg; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vcg (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vcg OWNER TO postgres;

--
-- Name: vci; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vci (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vci OWNER TO postgres;

--
-- Name: vgc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vgc (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vgc OWNER TO postgres;

--
-- Name: vhc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vhc (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vhc OWNER TO postgres;

--
-- Name: vhm; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vhm (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vhm OWNER TO postgres;

--
-- Name: vib; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vib (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vib OWNER TO postgres;

--
-- Name: vic; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vic (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vic OWNER TO postgres;

--
-- Name: vix; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vix (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vix OWNER TO postgres;

--
-- Name: vjc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vjc (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vjc OWNER TO postgres;

--
-- Name: vnd; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vnd (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vnd OWNER TO postgres;

--
-- Name: vnm; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vnm (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vnm OWNER TO postgres;

--
-- Name: vpb; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vpb (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vpb OWNER TO postgres;

--
-- Name: vpi; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vpi (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vpi OWNER TO postgres;

--
-- Name: vpl; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vpl (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vpl OWNER TO postgres;

--
-- Name: vre; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vre (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vre OWNER TO postgres;

--
-- Name: vsc; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vsc (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vsc OWNER TO postgres;

--
-- Name: vtp; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE public.vtp (
    symbol character varying(255),
    "time" date NOT NULL,
    open double precision,
    high double precision,
    low double precision,
    close double precision,
    volume bigint,
    percent_change double precision
);


ALTER TABLE public.vtp OWNER TO postgres;

--
-- Data for Name: acb; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.acb (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
ACB	2025-12-26	23.9	24.05	23.65	23.9	7831900	0
ACB	2025-12-29	23.85	24.1	23.8	24	3770400	0.42
ACB	2025-12-30	24	24.2	23.9	24.1	5337300	0.42
ACB	2025-12-31	24.15	24.25	23.9	24	5932900	-0.41
ACB	2026-01-05	24	24.1	23.85	24	8986200	0
ACB	2026-01-06	24.05	24.1	23.75	24	12736300	0
ACB	2026-01-07	24.2	24.75	24.1	24.65	22731900	2.71
ACB	2026-01-08	24.75	25.05	24.5	24.55	24925700	-0.41
ACB	2026-01-09	24.7	24.95	24.5	24.55	16460000	0
ACB	2026-01-12	24.65	25.5	24.65	25.5	22302500	3.87
ACB	2026-01-13	25.7	25.8	24.9	24.9	24138300	-2.35
ACB	2026-01-14	25	25.15	24.5	24.65	19456100	-1
ACB	2026-01-15	24.85	24.9	24.55	24.9	14169900	1.01
ACB	2026-01-16	25	25.1	24.65	24.85	10941400	-0.2
ACB	2026-01-19	24.9	25.15	24.75	25.1	10927600	1.01
ACB	2026-01-20	25.1	25.3	25	25.05	18384900	-0.2
ACB	2026-01-21	25	25.15	24.75	24.85	12924400	-0.8
ACB	2026-01-22	24.9	25.05	24.8	24.85	10084800	0
ACB	2026-01-23	25	25.1	24.9	25.05	11781000	0.8
ACB	2026-01-26	25.05	25.1	24.65	24.8	13742900	-1
ACB	2026-01-27	24.6	24.8	24.45	24.7	12471600	-0.4
ACB	2026-01-28	24.45	24.5	23.85	23.9	32294900	-3.24
ACB	2026-01-29	24	24.15	23.75	23.8	20344800	-0.42
ACB	2026-01-30	23.85	24.1	23.8	24.1	15667300	1.26
\.


--
-- Data for Name: anv; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.anv (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
ANV	2025-12-26	26.6	27.05	26.35	27	918600	0
ANV	2025-12-29	27	27.05	26.4	26.45	708700	-2.04
ANV	2025-12-30	26.55	26.7	26.4	26.45	450500	0
ANV	2025-12-31	26.65	26.65	25.7	25.7	1489200	-2.84
ANV	2026-01-05	25.7	26.05	24.85	25.35	1864500	-1.36
ANV	2026-01-06	25.35	25.7	25.25	25.4	904000	0.2
ANV	2026-01-07	25.65	26	25.45	25.65	770400	0.98
ANV	2026-01-08	25.95	25.95	25	25	1372600	-2.53
ANV	2026-01-09	25.15	25.4	24.8	24.95	1589100	-0.2
ANV	2026-01-12	24.95	25.55	24.4	25.2	1214300	1
ANV	2026-01-13	25.45	26.75	25.4	26.6	1923600	5.56
ANV	2026-01-14	26.6	27.35	26.3	26.9	2013600	1.13
ANV	2026-01-15	27.1	27.15	26.4	26.4	952400	-1.86
ANV	2026-01-16	26.65	26.7	26.15	26.25	1181000	-0.57
ANV	2026-01-19	26.35	27.75	26.25	27.6	1815600	5.14
ANV	2026-01-20	27.8	27.95	26.9	27	1526900	-2.17
ANV	2026-01-21	26.8	27.1	26	26.2	1300800	-2.96
ANV	2026-01-22	26.75	27.6	26.4	26.8	1414500	2.29
ANV	2026-01-23	27.9	28.45	26.9	26.9	2829000	0.37
ANV	2026-01-26	27.1	27.3	26.6	27.15	1064100	0.93
ANV	2026-01-27	27.15	27.15	26.45	26.7	1180900	-1.66
ANV	2026-01-28	26.85	27.4	26.75	27.05	1480200	1.31
ANV	2026-01-29	27.1	28.9	27.1	28.55	5703600	5.55
ANV	2026-01-30	28.6	29.25	28.4	28.55	1923800	0
\.


--
-- Data for Name: bcm; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.bcm (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
BCM	2025-12-26	59.77	59.77	57.51	58.59	216636	0
BCM	2025-12-29	58.49	62.62	57.8	59.47	245284	1.5
BCM	2025-12-30	60.16	60.45	58.49	60.45	160198	1.65
BCM	2025-12-31	59.47	60.75	59.47	60.75	148502	0.5
BCM	2026-01-05	60.55	60.55	59.47	60.06	140934	-1.14
BCM	2026-01-06	59.96	63.4	59.57	62.72	773557	4.43
BCM	2026-01-07	62.72	67.04	61.44	67.04	2249476	6.89
BCM	2026-01-08	70.38	70.48	66.84	67.04	2161296	0
BCM	2026-01-09	66.84	69.99	66.35	69.79	2150832	4.1
BCM	2026-01-12	69.79	72.74	67.83	68.81	1176194	-1.4
BCM	2026-01-13	68.81	68.81	65.86	67.43	2483710	-2.01
BCM	2026-01-14	68.81	72.15	67.73	72.15	2649024	7
BCM	2026-01-15	74.31	77.17	72.74	76.58	3741284	6.14
BCM	2026-01-16	73.72	78.15	73.72	76.18	2216007	-0.52
BCM	2026-01-19	74.71	79.62	74.12	77.46	2065202	1.68
BCM	2026-01-20	75.89	82.28	75.79	78.64	1745759	1.52
BCM	2026-01-21	76.87	77.76	74.71	75	1532919	-4.63
BCM	2026-01-22	74.12	75.79	72.25	73.23	2004246	-2.36
BCM	2026-01-23	72.25	72.74	68.12	68.12	1972111	-6.98
BCM	2026-01-26	67.53	70.19	66.84	66.94	1353714	-1.73
BCM	2026-01-27	66.06	67.63	63.5	65.96	1911474	-1.46
BCM	2026-01-28	66.75	67.53	64.48	64.88	1244185	-1.64
BCM	2026-01-29	64.88	65.86	63.99	64.88	1123719	0
BCM	2026-01-30	65.86	68.12	65.76	66.94	1825920	3.18
\.


--
-- Data for Name: bid; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.bid (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
BID	2025-12-26	38.45	38.8	38	38.8	2657200	0
BID	2025-12-29	38.8	38.9	38.6	38.8	1644300	0
BID	2025-12-30	38.8	39.85	38.65	39.4	5379000	1.55
BID	2025-12-31	39.5	39.55	38.9	38.9	1630700	-1.27
BID	2026-01-05	38.85	38.9	37.9	38.45	3885400	-1.16
BID	2026-01-06	38.3	38.9	37.8	38.9	4854900	1.17
BID	2026-01-07	39.5	41.4	39.5	40.95	13409700	5.27
BID	2026-01-08	41.3	43.8	40.9	43.05	27506700	5.13
BID	2026-01-09	43.8	46.05	43.7	46.05	22918600	6.97
BID	2026-01-12	47.9	49.25	47	49.25	7783600	6.95
BID	2026-01-13	49.35	51.6	48.6	51	21869900	3.55
BID	2026-01-14	50.8	54.5	50.2	54.5	17646800	6.86
BID	2026-01-15	52.6	54	50.7	50.7	26455700	-6.97
BID	2026-01-16	51	54.2	50.4	51.1	13973800	0.79
BID	2026-01-19	51.1	52.6	50.5	52	9409600	1.76
BID	2026-01-20	52.9	55.3	52.2	53.2	15547500	2.31
BID	2026-01-21	52.6	53.9	51.1	53	11935000	-0.38
BID	2026-01-22	53.5	56.6	51.6	52	18089900	-1.89
BID	2026-01-23	51.8	52.8	50.7	50.8	10080300	-2.31
BID	2026-01-26	51	52.6	50.8	52.5	11947600	3.35
BID	2026-01-27	52.5	52.8	49.5	52.6	12478800	0.19
BID	2026-01-28	53.4	55.9	51.4	51.9	17897800	-1.33
BID	2026-01-29	51.8	52.8	51	51.9	7082200	0
BID	2026-01-30	52.3	54.5	51.9	53.9	13696100	3.85
\.


--
-- Data for Name: bmp; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.bmp (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
BMP	2025-12-26	174.2	177.3	174.2	176.6	113700	0
BMP	2025-12-29	177	178.7	176.2	177	60500	0.23
BMP	2025-12-30	176	176.4	173	175.3	90700	-0.96
BMP	2025-12-31	175.7	175.7	174	175	57200	-0.17
BMP	2026-01-05	175	176.8	172.8	174.5	111500	-0.29
BMP	2026-01-06	176.4	176.4	172.8	173.5	125600	-0.57
BMP	2026-01-07	173.1	177.7	173.1	177.2	107100	2.13
BMP	2026-01-08	176.7	177	173	174.4	227300	-1.58
BMP	2026-01-09	174.6	175.5	173.5	175.1	96500	0.4
BMP	2026-01-12	175.1	180	173.2	174.7	110800	-0.23
BMP	2026-01-13	175.1	176.2	174	175.8	91800	0.63
BMP	2026-01-14	174.4	176.9	173.2	175	252600	-0.46
BMP	2026-01-15	175	177.8	174	174	147800	-0.57
BMP	2026-01-16	175	175.3	173.5	173.5	111900	-0.29
BMP	2026-01-19	175.8	176	169.2	169.8	361700	-2.13
BMP	2026-01-20	170	174.9	163.1	163.1	558300	-3.95
BMP	2026-01-21	163.1	165.5	157.5	160.1	374800	-1.84
BMP	2026-01-22	161.1	163	159.6	159.7	233500	-0.25
BMP	2026-01-23	162	162	157.9	158	169000	-1.06
BMP	2026-01-26	159.6	159.6	155	155.1	238500	-1.84
BMP	2026-01-27	155.1	163.9	155.1	162.9	402500	5.03
BMP	2026-01-28	163.5	164.6	159	160.5	140100	-1.47
BMP	2026-01-29	162.9	162.9	158	159.8	77200	-0.44
BMP	2026-01-30	159.8	163	157.5	163	289300	2
\.


--
-- Data for Name: bsi; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.bsi (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
BSI	2025-12-26	38	38.1	37.05	37.6	456100	0
BSI	2025-12-29	37.65	38.6	37.5	37.6	247000	0
BSI	2025-12-30	37.6	38.4	37.5	38.2	512200	1.6
BSI	2025-12-31	38.05	38.2	37.6	37.6	161600	-1.57
BSI	2026-01-05	37.6	37.8	37.05	37.75	482300	0.4
BSI	2026-01-06	37.75	37.85	37.1	37.6	292500	-0.4
BSI	2026-01-07	37.7	38.45	37.2	38.3	913800	1.86
BSI	2026-01-08	39.4	40.5	38.5	39.6	1815500	3.39
BSI	2026-01-09	40.3	40.45	39.5	39.95	931600	0.88
BSI	2026-01-12	40.95	42.7	40.3	42.7	2768600	6.88
BSI	2026-01-13	43.55	43.6	42.05	42.7	1267000	0
BSI	2026-01-14	42.8	43	41.15	42.2	1778000	-1.17
BSI	2026-01-15	41.3	42.15	41.05	41.3	989500	-2.13
BSI	2026-01-16	41.25	42.6	41	41	717100	-0.73
BSI	2026-01-19	41.2	41.75	40.8	41	568200	0
BSI	2026-01-20	41.1	41.6	39.95	40	1037700	-2.44
BSI	2026-01-21	40	40	38.35	38.8	1138300	-3
BSI	2026-01-22	39.1	39.95	38.85	39.4	490100	1.55
BSI	2026-01-23	39.6	40.15	39	39.05	565400	-0.89
BSI	2026-01-26	39.4	39.6	38.1	38.45	538800	-1.54
BSI	2026-01-27	38.5	38.75	38	38.55	252000	0.26
BSI	2026-01-28	38.9	38.9	38.15	38.3	443000	-0.65
BSI	2026-01-29	38.8	39.15	38.4	39.1	431000	2.09
BSI	2026-01-30	39.5	39.5	38.85	39.2	328600	0.26
\.


--
-- Data for Name: bsr; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.bsr (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
BSR	2025-12-26	15.9	16.3	15.7	16.25	11687700	0
BSR	2025-12-29	16.4	17.15	16.25	16.7	17508200	2.77
BSR	2025-12-30	16.9	16.9	16.2	16.2	7348700	-2.99
BSR	2025-12-31	16.3	16.35	16.1	16.1	5363600	-0.62
BSR	2026-01-05	16.6	16.95	16.15	16.25	16538300	0.93
BSR	2026-01-06	16.2	17.1	16.2	16.8	20485800	3.38
BSR	2026-01-07	16.9	17.95	16.75	17.95	31374900	6.85
BSR	2026-01-08	18.85	19.2	18.25	18.6	44567300	3.62
BSR	2026-01-09	18.7	19.9	18.7	19.7	38812900	5.91
BSR	2026-01-12	20.1	20.85	19.05	19.6	34088000	-0.51
BSR	2026-01-13	19.65	20.95	19.5	20.95	46664000	6.89
BSR	2026-01-14	21.6	21.9	20.4	21.4	38293200	2.15
BSR	2026-01-15	21	21.6	20.5	20.7	29699000	-3.27
BSR	2026-01-16	20.85	21.65	20	20.15	32240900	-2.66
BSR	2026-01-19	20.25	21.1	20.05	20.8	22521200	3.23
BSR	2026-01-20	21	21.25	20.7	20.75	23609200	-0.24
BSR	2026-01-21	20.45	22.2	20.3	22.2	40775000	6.99
BSR	2026-01-22	23.15	23.2	22	22.2	31658700	0
BSR	2026-01-23	22.25	22.35	20.65	20.65	40730800	-6.98
BSR	2026-01-26	20.9	21.55	20.55	20.85	24331600	0.97
BSR	2026-01-27	20.9	22	20.65	21.5	22591500	3.12
BSR	2026-01-28	22	23	21.5	21.9	41660200	1.86
BSR	2026-01-29	22	22.05	21.1	21.85	16709600	-0.23
BSR	2026-01-30	22.5	22.9	22.05	22.4	22201400	2.52
\.


--
-- Data for Name: bvh; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.bvh (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
BVH	2025-12-26	54.5	55.9	52.8	55.9	715000	0
BVH	2025-12-29	55.9	59.8	55.5	59.8	1306800	6.98
BVH	2025-12-30	60	60	58.2	59.4	452900	-0.67
BVH	2025-12-31	59.1	59.5	56.8	56.8	625000	-4.38
BVH	2026-01-05	56.9	59	56.2	57.3	601100	0.88
BVH	2026-01-06	57.3	61.3	57.3	61.3	920400	6.98
BVH	2026-01-07	63	65.5	63	65.5	1593200	6.85
BVH	2026-01-08	70	70	66	66.2	3021300	1.07
BVH	2026-01-09	69	69.4	66.6	67.5	1864400	1.96
BVH	2026-01-12	67.9	70	64.1	66	1804600	-2.22
BVH	2026-01-13	65.1	68	64.8	67.9	1210600	2.88
BVH	2026-01-14	68.6	72	67.2	68.5	2486700	0.88
BVH	2026-01-15	68.7	73.2	67.3	73.2	2122500	6.86
BVH	2026-01-16	78	78	73.5	73.7	2961800	0.68
BVH	2026-01-19	73.7	73.7	70.2	71.3	1369600	-3.26
BVH	2026-01-20	72	75.9	70.8	73	1698100	2.38
BVH	2026-01-21	71.3	75.4	71.3	73.9	1271800	1.23
BVH	2026-01-22	74.8	78.5	73	76	1644300	2.84
BVH	2026-01-23	75.2	75.2	72.1	72.6	1249000	-4.47
BVH	2026-01-26	71.1	71.1	67.6	67.8	1937800	-6.61
BVH	2026-01-27	68.3	70.8	66.9	69	1368400	1.77
BVH	2026-01-28	70.2	70.4	67.4	68.8	964300	-0.29
BVH	2026-01-29	68.8	69.3	66.6	69.1	748000	0.44
BVH	2026-01-30	69.3	71	68.5	69.8	707100	1.01
\.


--
-- Data for Name: bwe; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.bwe (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
BWE	2025-12-26	43.15	43.64	40.82	40.82	555799	0
BWE	2025-12-29	41.79	43.64	41.79	43.64	324236	6.91
BWE	2025-12-30	46.65	46.65	43.88	45.58	567657	4.45
BWE	2025-12-31	45.53	46.07	43.98	46.07	96058	1.08
BWE	2026-01-05	45.78	45.78	44.61	44.8	132012	-2.76
BWE	2026-01-06	45.14	45.14	43.93	44.03	100519	-1.72
BWE	2026-01-07	44.66	44.66	44.03	44.03	134794	0
BWE	2026-01-08	43.83	44.08	43.4	43.4	322128	-1.43
BWE	2026-01-09	43.4	44.66	43.35	43.59	97721	0.44
BWE	2026-01-12	44.22	44.22	42.76	43.88	180554	0.67
BWE	2026-01-13	44.03	45.68	43.74	45.68	218192	4.1
BWE	2026-01-14	45.63	45.63	45	45.53	114491	-0.33
BWE	2026-01-15	45.05	45.97	45.05	45.73	79617	0.44
BWE	2026-01-16	46.46	46.46	45.24	45.53	83372	-0.44
BWE	2026-01-19	45.68	45.87	45.19	45.58	129321	0.11
BWE	2026-01-20	45.58	45.87	45.39	45.68	284423	0.22
BWE	2026-01-21	42.86	45.53	42.86	45.19	703668	-1.07
BWE	2026-01-22	45.24	45.68	44.51	45.05	42569	-0.31
BWE	2026-01-23	44.8	44.8	43.83	44.08	132014	-2.15
BWE	2026-01-26	44.61	45.1	43.64	43.64	175399	-1
BWE	2026-01-27	44.56	44.56	43.35	44.03	189591	0.89
BWE	2026-01-28	44.03	44.71	43.78	44.03	121455	0
BWE	2026-01-29	44.42	45.1	44.32	45.1	128113	2.43
BWE	2026-01-30	45.39	45.68	44.71	45	129156	-0.22
\.


--
-- Data for Name: cii; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.cii (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
CII	2025-12-26	22.4	22.95	21.5	22.6	21648600	0
CII	2025-12-29	22.75	22.95	22.25	22.8	8177100	0.88
CII	2025-12-30	22.8	22.85	21.9	21.9	12196100	-3.95
CII	2025-12-31	21.9	22.1	20.9	20.9	28224500	-4.57
CII	2026-01-05	20.95	21.8	20.9	21	16836300	0.48
CII	2026-01-06	21.3	21.5	19.85	19.9	26600900	-5.24
CII	2026-01-07	20.05	20.45	19.5	19.95	19006700	0.25
CII	2026-01-08	20.3	20.8	19.6	19.7	20548700	-1.25
CII	2026-01-09	19.8	19.95	18.35	18.35	47257200	-6.85
CII	2026-01-12	18.15	19.35	17.45	18.8	27577900	2.45
CII	2026-01-13	19.3	19.8	18.9	19.05	21403500	1.33
CII	2026-01-14	19.05	19.5	18.55	18.55	21311100	-2.62
CII	2026-01-15	18.55	19.3	18.45	19.05	17994800	2.7
CII	2026-01-16	19.35	19.35	18.6	18.65	15897400	-2.1
CII	2026-01-19	18.4	19.25	18.4	18.9	15228200	1.34
CII	2026-01-20	19.05	19.1	18.5	18.5	13648100	-2.12
CII	2026-01-21	18.45	18.8	17.75	17.9	16385200	-3.24
CII	2026-01-22	18	19.15	17.9	19.15	17824300	6.98
CII	2026-01-23	19.45	19.45	18.6	18.6	18645000	-2.87
CII	2026-01-26	18.7	19	17.8	17.95	13575200	-3.49
CII	2026-01-27	18.05	18.2	17.25	17.25	13089500	-3.9
CII	2026-01-28	17.35	18.45	16.9	18.25	19114500	5.8
CII	2026-01-29	18.65	18.8	18.05	18.05	8512200	-1.1
CII	2026-01-30	18.2	18.8	18.2	18.45	17055900	2.22
\.


--
-- Data for Name: cmg; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.cmg (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
CMG	2025-12-26	31.82	31.86	31	31.73	313420	0
CMG	2025-12-29	31.82	32.23	31.64	32.18	284725	1.42
CMG	2025-12-30	32.36	32.64	32.14	32.41	397677	0.71
CMG	2025-12-31	32.59	32.68	32.32	32.36	267021	-0.15
CMG	2026-01-05	32.36	32.36	31.64	31.68	283939	-2.1
CMG	2026-01-06	31.68	32.14	31.41	31.77	357868	0.28
CMG	2026-01-07	32.05	32.91	31.91	32.73	369293	3.02
CMG	2026-01-08	32.73	33.64	32.73	32.95	514233	0.67
CMG	2026-01-09	32.95	34.5	32.95	33.68	462733	2.22
CMG	2026-01-12	34.36	34.36	33.59	33.59	444677	-0.27
CMG	2026-01-13	33.68	33.91	33.36	33.45	469655	-0.42
CMG	2026-01-14	34.25	34.4	34	34.1	438358	1.94
CMG	2026-01-15	34.5	36.45	34.1	36.45	1335014	6.89
CMG	2026-01-16	37	39	36.5	37.9	1599138	3.98
CMG	2026-01-19	38.85	38.85	37	37	688124	-2.37
CMG	2026-01-20	37.6	37.9	36.5	37.15	1015632	0.41
CMG	2026-01-21	36.9	37.8	36.15	37.65	709341	1.35
CMG	2026-01-22	37.95	38.5	37.15	37.15	453416	-1.33
CMG	2026-01-23	37.2	37.2	36	36	441917	-3.1
CMG	2026-01-26	35.6	36	34.65	34.95	576621	-2.92
CMG	2026-01-27	34.8	35.8	34.8	35.25	302700	0.86
CMG	2026-01-28	35.3	36.5	35.3	35.55	399600	0.85
CMG	2026-01-29	35.75	36.7	35.55	35.55	385200	0
CMG	2026-01-30	35.65	36.45	35.55	35.8	620400	0.7
\.


--
-- Data for Name: company_info; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.company_info (symbol, icb_name2, listing_date, ceo_name) FROM stdin;
ACB	Ngân hàng	2020-12-09	Mr. Từ Tiến Phát
ANV	Thực phẩm và đồ uống	2007-12-07	Mr. Doãn Tới 
BCM	Bất động sản	2020-08-31	Mr. Nguyễn Văn Hùng
BID	Ngân hàng	2014-01-24	Mr. Phan Đức Tú
BMP	Xây dựng và Vật liệu	2006-07-11	Mr. Niwat Athiwattananont
BSI	Dịch vụ tài chính	2011-07-19	Mr. Nguyễn Duy Viễn
BSR	Dầu khí	2025-01-17	Mr. Nguyễn Việt Tháng
BVH	Bảo hiểm	2009-06-25	Mr. Đỗ Trường Minh
BWE	Điện, nước & xăng dầu khí đốt	2017-07-20	Mr. Nguyễn Văn Thiền
CII	Xây dựng và Vật liệu	2006-05-18	Mr. Lê Quốc Bình
CMG	Công nghệ Thông tin	2010-01-22	Mr. Nguyễn Trung Chính
CTD	Xây dựng và Vật liệu	2010-01-20	Mr. Bolat Duisenov
CTG	Ngân hàng	2009-07-16	Mr. Trần Minh Bình
CTR	Xây dựng và Vật liệu	2022-02-23	Mr. Phạm Đình Trường
CTS	Dịch vụ tài chính	2017-06-20	Mr. Trần Phúc Vinh
DBC	Thực phẩm và đồ uống	2019-07-26	Mr. Nguyễn Như So
DCM	Hóa chất	2015-03-31	Mr. Văn Tiến Thanh
DGC	Hóa chất	2020-07-28	Mr. Lưu Bách Đạt
DGW	Bán lẻ	2015-08-03	Mr.  Đoàn Hồng Việt
DIG	Bất động sản	2009-08-19	Mr. Nguyễn Hùng Cường
DPM	Hóa chất	2007-11-05	Mr. Phan Công Thành
DSE	Dịch vụ tài chính	2024-07-01	Ms. Nguyễn Ngọc Linh
DXG	Bất động sản	2009-12-22	Mr. Bùi Ngọc Đức
DXS	Bất động sản	2021-07-15	Mr. Trần Quốc Thịnh
EIB	Ngân hàng	2009-10-27	Ms, Phạm Thị Huyền Trang
EVF	Dịch vụ tài chính	2022-01-12	Mr. Mai Danh Hiền
FPT	Công nghệ Thông tin	2006-12-13	Mr. Trương Gia Bình
FRT	Bán lẻ	2018-04-26	Mr. Hoàng Trung Kiên
FTS	Dịch vụ tài chính	2017-01-13	Mr. Nguyễn Điệp Tùng
GAS	Điện, nước & xăng dầu khí đốt	2012-05-21	Mr. Phạm Văn Phong
GEE	Hàng & Dịch vụ Công nghiệp	2022-03-08	Mr. Nguyễn Trọng Trung
GEX	Hàng & Dịch vụ Công nghiệp	2018-01-18	Mr. Nguyễn Văn Tuấn
GMD	Hàng & Dịch vụ Công nghiệp	2002-04-22	Mr. Nguyễn Thanh Bình
GVR	Hóa chất	2020-03-17	Mr. Trần Công Kha
HAG	Thực phẩm và đồ uống	2008-12-22	Mr. Nguyễn Xuân Thắng
HCM	Dịch vụ tài chính	2009-05-19	Mr. Trịnh Hoài Giang
HDB	Ngân hàng	2018-01-05	Mr. Nguyễn Hữu Đặng
HDC	Bất động sản	2007-10-08	Mr. Đoàn Hữu Thuận
HDG	Bất động sản	2010-02-02	Mr. Nguyễn Trọng Minh
HHV	Xây dựng và Vật liệu	2022-01-20	Mr. Ngọ Trường Nam
HPG	Tài nguyên Cơ bản	2007-11-15	Mr. Nguyễn Việt Thắng
HSG	Tài nguyên Cơ bản	2008-12-05	Mr. Lê Phước Vũ
HT1	Xây dựng và Vật liệu	2007-11-13	Mr, Nguyễn Quốc Thắng
IMP	Y tế	2006-12-04	Ms. Trần Thị Đào
KBC	Bất động sản	2009-12-18	Mr. Đặng Thành Tâm
KDC	Thực phẩm và đồ uống	2005-12-12	Mr. Trần Kim Thành
KDH	Bất động sản	2010-02-01	Mr. Vương Văn Minh
KOS	Bất động sản	2017-12-08	Mr. Nguyễn Việt Cường
LPB	Ngân hàng	2020-11-09	Mr. Vũ Quốc Khánh
MBB	Ngân hàng	2011-11-01	Mr. Lưu Trung Thái
MSB	Ngân hàng	2020-12-23	Mr. Nguyễn Hoàng Linh
MSN	Thực phẩm và đồ uống	2009-11-05	Mr. Nguyễn Đăng Quang
MWG	Bán lẻ	2014-07-14	Mr. Vũ Đăng Linh
NAB	Ngân hàng	2024-03-08	Mr. Trần Ngô Phúc Vũ
NKG	Tài nguyên Cơ bản	2011-01-14	Mr. Võ Hoàng Vũ
NLG	Bất động sản	2013-04-08	Mr. Nguyễn Xuân Quang
NT2	Điện, nước & xăng dầu khí đốt	2015-06-12	Mr. Ngô Đức Nhân
NVL	Bất động sản	2016-12-28	Mr. Dương Văn Bắc
OCB	Ngân hàng	2021-01-28	Mr.Trịnh Văn Tuấn 
PAN	Thực phẩm và đồ uống	2010-12-15	Mr. Nguyễn Duy Hưng
PC1	Xây dựng và Vật liệu	2016-11-16	Mr. Vũ Ánh Dương
PDR	Bất động sản	2010-07-30	Mr. Nguyễn Văn Đạt
PHR	Hóa chất	2009-08-18	Mr. Huỳnh Kim Nhựt
PLX	Dầu khí	2017-04-21	Mr. Phạm Văn Thanh
PNJ	Hàng cá nhân & Gia dụng	2009-03-23	Ms. Cao Thị Ngọc Dung
POW	Điện, nước & xăng dầu khí đốt	2019-01-14	Mr. Lê Như Linh
PVD	Dầu khí	2006-12-05	Mr. Nguyễn Xuân Cường
PVT	Hàng & Dịch vụ Công nghiệp	2007-12-10	Mr. Nguyễn Duyên Hiếu
REE	Điện, nước & xăng dầu khí đốt	2000-07-28	Mr. Ashok Ramachandran
SAB	Thực phẩm và đồ uống	2016-12-06	Mr. Koh Poh Tiong
SBT	Thực phẩm và đồ uống	2008-02-25	Mrs. Đặng Huỳnh Ức My
SCS	Du lịch và Giải trí	2018-08-03	Mr. Nguyễn Quốc Khánh
SHB	Ngân hàng	2021-10-11	Ms. Ngô Thu Hà
SIP	Bất động sản	2023-08-08	Mr. Trần Mạnh Hùng
SJS	Bất động sản	2006-07-06	Mr. Bùi Quang Bách
SSB	Ngân hàng	2021-03-24	Mr. Lê Văn Tần
SSI	Dịch vụ tài chính	2007-10-29	Mr. Nguyễn Duy Hưng
STB	Ngân hàng	2006-07-12	Ms. Nguyễn Đức Thạch Diễm
SZC	Bất động sản	2019-01-15	Mr. Nguyễn Văn Tuấn
TCH	Bất động sản	2016-10-05	Ms. Hoàng Thị Huyên
TPB	Ngân hàng	2018-04-19	Mr. Đỗ Minh Phú
VCB	Ngân hàng	2009-06-30	Mr. Nguyễn Thanh Tùng
VCG	Xây dựng và Vật liệu	2020-12-29	Mr. Nguyễn Xuân Đông
VCI	Dịch vụ tài chính	2017-07-07	Ms. Tôn Minh Phương
VGC	Xây dựng và Vật liệu	2019-05-29	Mr. Nguyễn Anh Tuấn
VHC	Thực phẩm và đồ uống	2007-12-24	Ms. Trương Thị Lệ Khanh
VHM	Bất động sản	2018-05-17	Mr. Phạm Thiếu Hoa
VIB	Ngân hàng	2020-11-10	Mr. Hàn Ngọc Vũ 
VIC	Bất động sản	2007-09-19	Mr. Nguyễn Việt Quang
VIX	Dịch vụ tài chính	2021-01-08	Mr. Trương Ngọc Lân
VJC	Du lịch và Giải trí	2017-02-28	Ms. Nguyễn Thị Phương Thảo
VND	Dịch vụ tài chính	2017-08-18	Ms. Phạm Minh Hương
VNM	Thực phẩm và đồ uống	2006-01-19	Ms. Mai Kiều Liên
VPB	Ngân hàng	2017-08-17	Mr. Ngô Chí Dũng
VPI	Bất động sản	2018-06-29	Mr. Tô Như Toàn
VPL	Du lịch và Giải trí	2025-05-13	Ms. Nguyễn Thu Hằng
VRE	Bất động sản	2017-11-06	Ms. Phạm Thị Thu Hiền
VSC	Hàng & Dịch vụ Công nghiệp	2008-01-09	Mr. Tạ Công Thông 
VTP	Hàng & Dịch vụ Công nghiệp	2024-03-12	Mr. Phùng Văn Cường
\.


--
-- Data for Name: ctd; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ctd (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
CTD	2025-12-26	77.3	77.5	75	75.1	581349	0
CTD	2025-12-29	75.8	76.3	74.8	74.9	440745	-0.27
CTD	2025-12-30	75	75.3	69.7	73.3	1147843	-2.14
CTD	2025-12-31	73.6	76	72.9	76	687812	3.68
CTD	2026-01-05	76.3	76.3	73.4	74.6	642969	-1.84
CTD	2026-01-06	74.6	74.6	72.5	74	446123	-0.8
CTD	2026-01-07	74	74.3	72.5	72.5	837756	-2.03
CTD	2026-01-08	73.1	73.7	72.5	72.5	649000	0
CTD	2026-01-09	73	73.4	72.3	72.4	541800	-0.14
CTD	2026-01-12	72.4	75.4	72.4	75	785600	3.59
CTD	2026-01-13	75.1	78.5	75.1	77.1	1247400	2.8
CTD	2026-01-14	77.1	77.8	75.6	77.1	569900	0
CTD	2026-01-15	77.1	77.3	75	75.3	664800	-2.33
CTD	2026-01-16	75.5	77.7	75	76.6	767000	1.73
CTD	2026-01-19	76.6	78.9	76.4	77	507100	0.52
CTD	2026-01-20	78.2	78.2	75.2	75.2	898900	-2.34
CTD	2026-01-21	75.2	75.6	74	74.1	655100	-1.46
CTD	2026-01-22	74.9	76.5	74.5	75	469100	1.21
CTD	2026-01-23	75	76.8	74.6	74.6	406800	-0.53
CTD	2026-01-26	75.5	78	74.7	76.9	1174000	3.08
CTD	2026-01-27	76.6	80.1	76.6	77.3	852000	0.52
CTD	2026-01-28	78.9	78.9	76.1	76.5	626500	-1.03
CTD	2026-01-29	76.5	77.5	74.5	75.9	797200	-0.78
CTD	2026-01-30	75.9	77.6	75.6	76.8	511200	1.19
\.


--
-- Data for Name: ctg; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ctg (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
CTG	2025-12-26	34.9	35.25	34.25	35.2	10634744	0
CTG	2025-12-29	35	35.35	34.9	35.3	7200183	0.28
CTG	2025-12-30	35.35	35.95	35.2	35.65	8456300	0.99
CTG	2025-12-31	35.7	35.8	35.4	35.75	7920700	0.28
CTG	2026-01-05	35.8	36.1	34.55	35.5	10427300	-0.7
CTG	2026-01-06	35.4	36.2	34.8	36.2	14857900	1.97
CTG	2026-01-07	36.4	37.5	36.35	37.45	22144400	3.45
CTG	2026-01-08	37.55	39.5	37.1	38.2	31939500	2
CTG	2026-01-09	39	40.85	38.7	40.75	38922900	6.68
CTG	2026-01-12	42	43.5	41.35	41.5	37246900	1.84
CTG	2026-01-13	41.5	42.35	40	41.3	32378300	-0.48
CTG	2026-01-14	40.8	43.25	40.65	41.45	30568900	0.36
CTG	2026-01-15	40.5	41	39.6	40	28177000	-3.5
CTG	2026-01-16	40.7	41.55	39.5	39.6	21678400	-1
CTG	2026-01-19	39.65	40.35	39.55	40	13549500	1.01
CTG	2026-01-20	40.45	40.6	39.65	39.65	18904200	-0.88
CTG	2026-01-21	39.55	40.55	39	40.2	20390200	1.39
CTG	2026-01-22	40.4	41.3	39.85	39.9	21656100	-0.75
CTG	2026-01-23	40	40.05	38.85	39	17979800	-2.26
CTG	2026-01-26	39.05	39.15	38.05	38.2	17221600	-2.05
CTG	2026-01-27	38.2	38.65	38.1	38.25	12601700	0.13
CTG	2026-01-28	38.35	39.65	38.3	38.4	15308300	0.39
CTG	2026-01-29	38.5	38.9	38.1	38.3	10027200	-0.26
CTG	2026-01-30	38.2	39	37.8	38.75	28474900	1.17
\.


--
-- Data for Name: ctr; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ctr (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
CTR	2025-12-26	83.8	84	82.9	84	309400	0
CTR	2025-12-29	84	85.2	84	84.7	193600	0.83
CTR	2025-12-30	84	86	84	85.3	304800	0.71
CTR	2025-12-31	85.5	85.5	84.8	85.3	297100	0
CTR	2026-01-05	85.3	85.3	83.3	83.3	251900	-2.34
CTR	2026-01-06	83.5	85.1	83.3	85	281800	2.04
CTR	2026-01-07	85	89.9	85	89.1	979200	4.82
CTR	2026-01-08	89.5	92	89.5	89.7	1106300	0.67
CTR	2026-01-09	90.3	95.9	90.3	95.9	1878900	6.91
CTR	2026-01-12	102.6	102.6	102.6	102.6	667600	6.99
CTR	2026-01-13	109.4	109.4	98.3	101	3086000	-1.56
CTR	2026-01-14	105	108	102.4	108	3364900	6.93
CTR	2026-01-15	109.5	113	104	107	1658300	-0.93
CTR	2026-01-16	105	110.5	102.6	102.6	1925200	-4.11
CTR	2026-01-19	102.6	104.2	100.6	102.8	1325000	0.19
CTR	2026-01-20	103.6	107.9	103.1	103.6	1361000	0.78
CTR	2026-01-21	102	103	98.9	101.4	1449600	-2.12
CTR	2026-01-22	103.7	104.1	99	99	1169200	-2.37
CTR	2026-01-23	98.9	99	96	96	1222800	-3.03
CTR	2026-01-26	96.2	98	94.4	94.9	970500	-1.15
CTR	2026-01-27	95	98	93.1	97	681400	2.21
CTR	2026-01-28	99	99.9	96.1	96.7	927000	-0.31
CTR	2026-01-29	96.7	97.7	96	96	562800	-0.72
CTR	2026-01-30	96.1	98.7	96.1	96.7	613000	0.73
\.


--
-- Data for Name: cts; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.cts (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
CTS	2025-12-26	33.9	34.45	32.9	33.8	980100	0
CTS	2025-12-29	33.8	34.5	33.45	33.5	402000	-0.89
CTS	2025-12-30	33.55	33.85	33.5	33.6	432300	0.3
CTS	2025-12-31	33.95	33.95	33.15	33.15	488000	-1.34
CTS	2026-01-05	33.2	33.3	30.85	30.85	1610500	-6.94
CTS	2026-01-06	31.2	31.55	30.25	31.05	954800	0.65
CTS	2026-01-07	31.05	32.85	31.05	31.9	651000	2.74
CTS	2026-01-08	32.3	32.9	31.9	32.1	1197200	0.63
CTS	2026-01-09	32.55	32.6	32	32.1	935900	0
CTS	2026-01-12	32.6	34.3	32.6	34.3	1910000	6.85
CTS	2026-01-13	35.65	35.7	34.2	34.5	1643900	0.58
CTS	2026-01-14	34.6	34.7	33.7	33.95	1970100	-1.59
CTS	2026-01-15	33.95	34.4	33.65	33.65	1411800	-0.88
CTS	2026-01-16	33.7	34.95	33.3	33.8	1409800	0.45
CTS	2026-01-19	33.9	34.7	33.9	33.95	843800	0.44
CTS	2026-01-20	34.2	34.55	33.85	33.85	1600200	-0.29
CTS	2026-01-21	33.75	33.75	32	32.6	1930400	-3.69
CTS	2026-01-22	33.1	33.4	32.5	33	1113700	1.23
CTS	2026-01-23	33.1	33.7	32.8	32.85	727800	-0.45
CTS	2026-01-26	32.85	33.15	31.7	31.7	999500	-3.5
CTS	2026-01-27	31.7	32	31.3	31.45	738100	-0.79
CTS	2026-01-28	31.9	31.9	30.75	30.9	1177200	-1.75
CTS	2026-01-29	31	31.5	31	31.2	415900	0.97
CTS	2026-01-30	31.25	31.75	31.1	31.25	549500	0.16
\.


--
-- Data for Name: dbc; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.dbc (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
DBC	2025-12-26	27.8	28.05	27.1	27.55	6835800	0
DBC	2025-12-29	27.5	27.8	27.25	27.3	3197700	-0.91
DBC	2025-12-30	27.45	27.7	27.2	27.45	3911000	0.55
DBC	2025-12-31	27.5	27.55	27.05	27.05	3294800	-1.46
DBC	2026-01-05	27.15	27.3	26.4	26.5	7597100	-2.03
DBC	2026-01-06	26.7	26.9	26.45	26.7	4233700	0.75
DBC	2026-01-07	26.8	27.15	26.65	26.85	4849200	0.56
DBC	2026-01-08	27	27.2	26.55	26.55	5049200	-1.12
DBC	2026-01-09	26.75	26.8	25.75	26	8728200	-2.07
DBC	2026-01-12	26.35	27.4	26.3	27.3	4728900	5
DBC	2026-01-13	27.5	27.9	27.35	27.5	5858900	0.73
DBC	2026-01-14	27.8	29.25	27.6	28.75	19023900	4.55
DBC	2026-01-15	29.05	29.05	28.1	28.3	5287000	-1.57
DBC	2026-01-16	28.55	28.7	28	28	5276700	-1.06
DBC	2026-01-19	28.15	28.2	27.8	28	3958700	0
DBC	2026-01-20	28.25	28.55	27.8	27.85	5767600	-0.54
DBC	2026-01-21	27.95	28.25	27.55	28.25	6051000	1.44
DBC	2026-01-22	28.5	29.6	28.5	28.6	12470600	1.24
DBC	2026-01-23	28.9	28.9	28.1	28.15	5156700	-1.57
DBC	2026-01-26	28.35	28.8	27.25	27.55	10440100	-2.13
DBC	2026-01-27	27.7	29.2	27.5	28.55	12497000	3.63
DBC	2026-01-28	28.8	29.05	28	28.25	6870600	-1.05
DBC	2026-01-29	28.45	29.5	28.3	28.3	12944900	0.18
DBC	2026-01-30	28.6	28.6	28.1	28.15	8545700	-0.53
\.


--
-- Data for Name: dcm; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.dcm (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
DCM	2025-12-26	32	32.5	31.55	31.8	2252100	0
DCM	2025-12-29	32.1	32.9	32.1	32.7	1618300	2.83
DCM	2025-12-30	32.95	32.95	32.55	32.7	492800	0
DCM	2025-12-31	32.7	32.95	32.4	32.7	1353600	0
DCM	2026-01-05	32.8	33.65	32.6	33.35	2369000	1.99
DCM	2026-01-06	33.7	33.7	33	33.4	1192100	0.15
DCM	2026-01-07	33.5	34.15	33.3	34	2524200	1.8
DCM	2026-01-08	34.4	35.5	34.4	34.65	5787700	1.91
DCM	2026-01-09	34.85	35.5	34.5	34.65	2942900	0
DCM	2026-01-12	35.4	36.35	35.1	35.3	5198200	1.88
DCM	2026-01-13	35.45	35.9	34.7	35.65	3452700	0.99
DCM	2026-01-14	35.85	37.2	35.35	36.7	7092200	2.95
DCM	2026-01-15	36.7	36.9	35.6	35.7	3521000	-2.72
DCM	2026-01-16	35.9	36	35.2	35.2	2425100	-1.4
DCM	2026-01-19	35.2	35.5	35	35.1	1960300	-0.28
DCM	2026-01-20	35	35.75	34.9	35.2	3028600	0.28
DCM	2026-01-21	35	36.1	35	35.5	3355000	0.85
DCM	2026-01-22	35.7	37.95	35.7	37.95	10965800	6.9
DCM	2026-01-23	38.1	38.15	36	36.05	4146800	-5.01
DCM	2026-01-26	36.15	36.25	34.25	34.8	5677900	-3.47
DCM	2026-01-27	34.9	35.95	34.85	35.8	2905000	2.87
DCM	2026-01-28	36	36.6	35.5	36	3438300	0.56
DCM	2026-01-29	36	37.2	35.55	37	4027900	2.78
DCM	2026-01-30	37.1	37.5	36.55	36.6	4038100	-1.08
\.


--
-- Data for Name: dgc; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.dgc (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
DGC	2025-12-26	64	64	60.1	60.9	10121929	0
DGC	2025-12-29	60	63.5	58.2	61	6348695	0.16
DGC	2025-12-30	62.2	65.2	61.6	65.2	7237123	6.89
DGC	2025-12-31	67	68.8	66.1	68.5	11132279	5.06
DGC	2026-01-05	68.6	68.7	65	65.5	6505134	-4.38
DGC	2026-01-06	65.2	66.3	62.7	63	5489950	-3.82
DGC	2026-01-07	63.2	65.5	63.2	64.1	3215762	1.75
DGC	2026-01-08	65	65	62.7	62.8	5234400	-2.03
DGC	2026-01-09	62.9	63.8	62.6	62.7	4030000	-0.16
DGC	2026-01-12	63.9	64.3	62.6	63.7	4044400	1.59
DGC	2026-01-13	64	64.3	62.9	63.1	4373300	-0.94
DGC	2026-01-14	63.1	64.8	62.9	63.9	4895300	1.27
DGC	2026-01-15	64.3	67.9	63.8	67.3	6973100	5.32
DGC	2026-01-16	68	68.1	66	66	4181100	-1.93
DGC	2026-01-19	66	66.8	65.1	65.1	3348700	-1.36
DGC	2026-01-20	65.3	66	64.3	64.5	3021000	-0.92
DGC	2026-01-21	64.3	69	64	69	10749300	6.98
DGC	2026-01-22	71.4	73.8	70.2	73.8	9095300	6.96
DGC	2026-01-23	76	77.9	71.5	73.9	8651600	0.14
DGC	2026-01-26	73.8	73.8	68.8	68.8	7179700	-6.9
DGC	2026-01-27	67	68.8	66.1	67.5	4918700	-1.89
DGC	2026-01-28	67.7	69.4	67	68.1	3006800	0.89
DGC	2026-01-29	68.4	68.6	66.1	67.8	2642600	-0.44
DGC	2026-01-30	69	71	68.2	68.6	3461500	1.18
\.


--
-- Data for Name: dgw; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.dgw (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
DGW	2025-12-26	39.2	39.5	38.5	39.05	1186900	0
DGW	2025-12-29	38.9	40.3	38.85	40.1	1293100	2.69
DGW	2025-12-30	40.1	40.25	39.5	39.6	628000	-1.25
DGW	2025-12-31	39.6	39.85	39	39	887900	-1.52
DGW	2026-01-05	39.05	41.7	38.5	41.7	4123800	6.92
DGW	2026-01-06	42.15	42.45	41.5	42.2	1423400	1.2
DGW	2026-01-07	42.4	43.55	41.6	42.95	2287600	1.78
DGW	2026-01-08	43.05	43.05	41.2	41.2	2142500	-4.07
DGW	2026-01-09	41.6	42.85	40.8	42	1717700	1.94
DGW	2026-01-12	42.5	43.95	41.8	43.7	3045400	4.05
DGW	2026-01-13	44	44.3	42.95	43.45	2261600	-0.57
DGW	2026-01-14	43.6	45.9	43.45	44.1	4248800	1.5
DGW	2026-01-15	44.75	44.75	42.6	43	2218900	-2.49
DGW	2026-01-16	42.8	46	42.6	46	4786900	6.98
DGW	2026-01-19	46.7	46.8	45.05	46.15	2384400	0.33
DGW	2026-01-20	46.3	49.35	46.2	46.9	7172400	1.63
DGW	2026-01-21	47	47	45.05	45.85	2498100	-2.24
DGW	2026-01-22	46.2	47	44.8	45	2820300	-1.85
DGW	2026-01-23	45	46	44.55	45.75	1885300	1.67
DGW	2026-01-26	45.9	45.9	43.45	44.25	2735400	-3.28
DGW	2026-01-27	43.8	45.5	43.15	45.2	2075300	2.15
DGW	2026-01-28	45.2	45.95	44	44	2089400	-2.65
DGW	2026-01-29	44.1	47.05	44.1	47.05	4961300	6.93
DGW	2026-01-30	48	49.5	48	49	6248100	4.14
\.


--
-- Data for Name: dig; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.dig (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
DIG	2025-12-26	17.6	17.9	17.05	17.5	10865900	0
DIG	2025-12-29	17.6	17.75	17.3	17.4	6202200	-0.57
DIG	2025-12-30	17.45	17.5	17.05	17.05	8231200	-2.01
DIG	2025-12-31	17.05	17.2	16.75	16.75	9557000	-1.76
DIG	2026-01-05	17	17.5	16.95	17.3	10998600	3.28
DIG	2026-01-06	17.45	17.5	16.6	17	8531700	-1.73
DIG	2026-01-07	17.1	17.25	16.85	17.15	6508700	0.88
DIG	2026-01-08	17.15	17.5	16.8	17	16548000	-0.87
DIG	2026-01-09	16.95	17	15.85	16	30057000	-5.88
DIG	2026-01-12	16	16.85	15.6	16.6	17134500	3.75
DIG	2026-01-13	16.95	17.05	16.5	16.75	17243400	0.9
DIG	2026-01-14	16.75	17	16.3	16.7	15200400	-0.3
DIG	2026-01-15	16.45	17	16.3	16.5	13607500	-1.2
DIG	2026-01-16	16.6	16.65	16.1	16.15	12654800	-2.12
DIG	2026-01-19	16.15	16.5	16.1	16.25	5961900	0.62
DIG	2026-01-20	16.25	16.4	15.9	16.05	9453700	-1.23
DIG	2026-01-21	16.05	16.2	15.35	15.55	15787100	-3.12
DIG	2026-01-22	15.7	16.6	15.5	16.6	23064500	6.75
DIG	2026-01-23	16.6	16.6	16.05	16.1	7287100	-3.01
DIG	2026-01-26	16.2	16.3	15.35	15.55	10352300	-3.42
DIG	2026-01-27	15.6	15.85	15.4	15.6	6371600	0.32
DIG	2026-01-28	15.65	16.2	15.1	15.95	11728700	2.24
DIG	2026-01-29	16.1	16.2	15.85	16	3284600	0.31
DIG	2026-01-30	16.15	16.5	15.95	16.2	13135200	1.25
\.


--
-- Data for Name: dpm; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.dpm (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
DPM	2025-12-26	22.05	22.3	21.75	22.1	2237400	0
DPM	2025-12-29	22.45	22.95	22.2	22.7	3716400	2.71
DPM	2025-12-30	22.7	22.75	22.4	22.4	1881500	-1.32
DPM	2025-12-31	22.4	22.6	22.25	22.25	1429700	-0.67
DPM	2026-01-05	22.5	22.9	22.2	22.5	4704800	1.12
DPM	2026-01-06	22.65	23.1	22.65	22.85	4493900	1.56
DPM	2026-01-07	23	23.6	22.8	23.45	5731500	2.63
DPM	2026-01-08	23.85	24.1	23.4	23.5	7920400	0.21
DPM	2026-01-09	23.45	23.95	23.45	23.75	6041200	1.06
DPM	2026-01-12	23.85	24.7	23.85	24.3	8618100	2.32
DPM	2026-01-13	24.4	24.65	23.75	24.4	8015500	0.41
DPM	2026-01-14	24.65	25.5	24.2	24.9	12596300	2.05
DPM	2026-01-15	24.95	24.95	24.3	24.3	5419400	-2.41
DPM	2026-01-16	24.4	24.5	24	24.1	4680200	-0.82
DPM	2026-01-19	24.2	24.45	24	24.2	2722200	0.41
DPM	2026-01-20	24.3	24.7	24.15	24.3	4862900	0.41
DPM	2026-01-21	24.15	24.7	24.1	24.65	7793300	1.44
DPM	2026-01-22	24.8	26.35	24.8	25.85	18893400	4.87
DPM	2026-01-23	25.85	25.85	24.4	24.55	6074700	-5.03
DPM	2026-01-26	24.55	24.6	22.85	23	11076900	-6.31
DPM	2026-01-27	23.1	23.7	23.05	23.3	3200400	1.3
DPM	2026-01-28	23.6	23.9	23.4	23.6	3524900	1.29
DPM	2026-01-29	23.85	24.05	23.4	23.9	3282800	1.27
DPM	2026-01-30	24.25	24.25	23.85	23.95	4599700	0.21
\.


--
-- Data for Name: dse; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.dse (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
DSE	2025-12-26	23.89	23.89	23.13	23.62	514282	0
DSE	2025-12-29	23.62	23.71	23.49	23.62	321798	0
DSE	2025-12-30	23.4	23.67	23.27	23.4	385978	-0.93
DSE	2025-12-31	23.4	23.54	23.36	23.4	335963	0
DSE	2026-01-05	23.4	23.45	21.79	21.92	877401	-6.32
DSE	2026-01-06	21.88	22.41	21.88	22.41	390824	2.24
DSE	2026-01-07	22.41	22.64	22.41	22.55	312482	0.62
DSE	2026-01-08	22.55	22.74	22.51	22.6	330407	0.22
DSE	2026-01-09	23.05	23.55	22.8	23	295015	1.77
DSE	2026-01-12	23	24.6	22.9	24.6	912977	6.96
DSE	2026-01-13	24.65	25.2	24.65	24.8	601097	0.81
DSE	2026-01-14	24.8	24.8	24.05	24.1	510537	-2.82
DSE	2026-01-15	24.1	24.25	23.95	24.1	323109	0
DSE	2026-01-16	24.2	24.35	23.6	24.1	563100	0
DSE	2026-01-19	24.1	24.3	23.95	24	286610	-0.41
DSE	2026-01-20	24	24.3	23.55	23.55	419600	-1.87
DSE	2026-01-21	23.55	23.55	22.6	23.1	353188	-1.91
DSE	2026-01-22	23.1	24.7	23.1	24.7	1151400	6.93
DSE	2026-01-23	26	26	25.1	25.2	718600	2.02
DSE	2026-01-26	25.2	25.25	24.5	25	279400	-0.79
DSE	2026-01-27	25	25.2	24.7	24.95	257500	-0.2
DSE	2026-01-28	24.95	25	24.45	24.55	330800	-1.6
DSE	2026-01-29	24.55	24.65	24.1	24.3	233900	-1.02
DSE	2026-01-30	24.3	24.6	24.25	24.55	290600	1.03
\.


--
-- Data for Name: dxg; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.dxg (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
DXG	2025-12-26	17.5	17.65	16.8	17.6	16044400	0
DXG	2025-12-29	17.7	17.75	17.3	17.65	6373200	0.28
DXG	2025-12-30	17.6	17.8	17.25	17.5	8075700	-0.85
DXG	2025-12-31	17.45	17.45	17.1	17.1	7797100	-2.29
DXG	2026-01-05	17.1	17.4	16.65	17	8383800	-0.58
DXG	2026-01-06	16.85	17	16.45	16.65	9547900	-2.06
DXG	2026-01-07	16.8	16.95	16.55	16.75	10864500	0.6
DXG	2026-01-08	16.9	17.1	16.4	16.5	15743200	-1.49
DXG	2026-01-09	16.5	16.65	15.4	15.5	33186300	-6.06
DXG	2026-01-12	15.55	16.35	15.3	16.1	20904200	3.87
DXG	2026-01-13	16.55	16.7	16.05	16.2	17185600	0.62
DXG	2026-01-14	16.05	16.55	15.75	15.9	17778500	-1.85
DXG	2026-01-15	15.7	16.45	15.65	15.95	17357500	0.31
DXG	2026-01-16	16.05	16.25	15.75	15.9	10846100	-0.31
DXG	2026-01-19	15.95	16.15	15.7	15.7	12191700	-1.26
DXG	2026-01-20	15.75	15.95	15.4	15.45	13120400	-1.59
DXG	2026-01-21	15.3	15.85	14.95	15.2	18304200	-1.62
DXG	2026-01-22	15.25	16.25	14.65	15.75	39056400	3.62
DXG	2026-01-23	15.8	15.9	15.2	15.3	9782400	-2.86
DXG	2026-01-26	15.25	15.4	14.55	14.65	18054700	-4.25
DXG	2026-01-27	14.7	15	14.6	15	7028500	2.39
DXG	2026-01-28	15	15.4	14.6	15.2	12649600	1.33
DXG	2026-01-29	15.3	15.45	15.1	15.2	4817800	0
DXG	2026-01-30	15.25	15.6	15.15	15.4	11584600	1.32
\.


--
-- Data for Name: dxs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.dxs (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
DXS	2025-12-26	9.15	9.15	8.83	9.15	1989400	0
DXS	2025-12-29	9.09	9.22	9	9.21	895100	0.66
DXS	2025-12-30	9.11	9.25	9.06	9.2	1000000	-0.11
DXS	2025-12-31	9.19	9.19	9	9.07	1958000	-1.41
DXS	2026-01-05	9.07	9.1	8.7	8.94	2918500	-1.43
DXS	2026-01-06	8.93	8.94	8.49	8.79	2173700	-1.68
DXS	2026-01-07	8.72	8.85	8.6	8.81	2449600	0.23
DXS	2026-01-08	8.8	8.83	8.67	8.77	2935200	-0.45
DXS	2026-01-09	8.68	8.7	8.16	8.16	4351600	-6.96
DXS	2026-01-12	8.11	8.5	7.93	8.4	4356900	2.94
DXS	2026-01-13	8.58	8.58	8.38	8.4	1986600	0
DXS	2026-01-14	8.32	8.44	8.2	8.2	2599800	-2.38
DXS	2026-01-15	8.21	8.38	8.13	8.2	2386000	0
DXS	2026-01-16	8.18	8.33	8.12	8.17	2913900	-0.37
DXS	2026-01-19	8.17	8.2	8.08	8.16	2460200	-0.12
DXS	2026-01-20	8.22	8.22	7.89	7.89	3492700	-3.31
DXS	2026-01-21	7.89	7.98	7.71	7.74	1814300	-1.9
DXS	2026-01-22	7.83	8.2	7.59	8.07	5103400	4.26
DXS	2026-01-23	8.1	8.1	7.81	7.81	1661800	-3.22
DXS	2026-01-26	7.96	7.96	7.4	7.41	2374900	-5.12
DXS	2026-01-27	7.41	7.6	7.41	7.43	1681400	0.27
DXS	2026-01-28	7.47	7.48	7.18	7.26	3094700	-2.29
DXS	2026-01-29	7.26	7.4	7.25	7.36	969200	1.38
DXS	2026-01-30	7.37	7.55	7.36	7.45	1490900	1.22
\.


--
-- Data for Name: eib; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.eib (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
EIB	2025-12-26	21.95	21.95	21	21.55	8216700	0
EIB	2025-12-29	21.55	21.75	21.3	21.3	3121400	-1.16
EIB	2025-12-30	21.35	21.65	21.3	21.65	4800700	1.64
EIB	2025-12-31	21.65	21.75	21.3	21.3	4140300	-1.62
EIB	2026-01-05	21.5	21.5	20.5	20.95	7289700	-1.64
EIB	2026-01-06	20.95	21.5	20.6	21.2	5987500	1.19
EIB	2026-01-07	21.5	22	21.4	21.75	7080600	2.59
EIB	2026-01-08	21.7	22.2	21.45	21.85	9955200	0.46
EIB	2026-01-09	21.95	22.05	21.3	21.3	7257700	-2.52
EIB	2026-01-12	21.3	22.15	20.95	22	10663900	3.29
EIB	2026-01-13	22.3	23.5	21.9	23.5	34166600	6.82
EIB	2026-01-14	23.5	24.2	22.95	23.05	15903000	-1.91
EIB	2026-01-15	22.95	23.35	22.4	23.1	16863200	0.22
EIB	2026-01-16	23.45	23.45	22.7	23	9040500	-0.43
EIB	2026-01-19	23	23.4	22.7	23.1	7030000	0.43
EIB	2026-01-20	23.2	23.4	22.85	22.9	8531500	-0.87
EIB	2026-01-21	22.55	22.9	22.15	22.2	9812500	-3.06
EIB	2026-01-22	22.25	22.95	22.25	22.4	5322900	0.9
EIB	2026-01-23	22.5	22.95	22.4	22.55	6637000	0.67
EIB	2026-01-26	22.6	22.9	21.3	21.6	9672700	-4.21
EIB	2026-01-27	21.7	21.8	21	21	6750500	-2.78
EIB	2026-01-28	21.05	21.35	20.7	20.85	7699900	-0.71
EIB	2026-01-29	21.3	21.3	20.85	20.95	3257800	0.48
EIB	2026-01-30	21.2	21.2	20.7	20.85	9228500	-0.48
\.


--
-- Data for Name: evf; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.evf (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
EVF	2025-12-26	11.2	11.25	10.85	11.15	3814200	0
EVF	2025-12-29	11.15	11.25	11.05	11.05	1503400	-0.9
EVF	2025-12-30	11.1	11.15	11	11.05	2145400	0
EVF	2025-12-31	11.1	11.2	11.05	11.05	2168600	0
EVF	2026-01-05	11.05	11.1	10.7	10.75	3594600	-2.71
EVF	2026-01-06	10.85	11	10.6	10.9	3060500	1.4
EVF	2026-01-07	10.95	11.3	10.95	11.2	4159700	2.75
EVF	2026-01-08	11.4	11.4	11.15	11.15	4296700	-0.45
EVF	2026-01-09	11.2	11.25	11	11.05	3164800	-0.9
EVF	2026-01-12	11.05	11.55	11.05	11.5	6229600	4.07
EVF	2026-01-13	11.6	12.15	11.45	11.7	6387800	1.74
EVF	2026-01-14	11.75	11.85	11.45	11.55	4619200	-1.28
EVF	2026-01-15	11.55	11.9	11.5	11.7	5726200	1.3
EVF	2026-01-16	11.85	11.85	11.55	11.55	3508000	-1.28
EVF	2026-01-19	11.6	11.7	11.5	11.55	3156600	0
EVF	2026-01-20	11.6	11.7	11.5	11.55	3254500	0
EVF	2026-01-21	11.5	11.5	11.15	11.3	5925500	-2.16
EVF	2026-01-22	11.4	12.05	11.35	12.05	6976700	6.64
EVF	2026-01-23	12.25	12.45	11.75	11.75	8941300	-2.49
EVF	2026-01-26	11.8	11.8	11.25	11.3	4263700	-3.83
EVF	2026-01-27	11.4	11.5	11.2	11.5	3658400	1.77
EVF	2026-01-28	11.5	11.75	11.4	11.55	4361100	0.43
EVF	2026-01-29	11.75	12	11.6	11.65	3386600	0.87
EVF	2026-01-30	11.65	11.75	11.5	11.55	4281600	-0.86
\.


--
-- Data for Name: fpt; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.fpt (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
FPT	2025-12-26	92.7	93	91	92.5	6612400	0
FPT	2025-12-29	92.5	94.9	92.5	94.3	6300900	1.95
FPT	2025-12-30	94.5	96.6	94.3	96.5	7271200	2.33
FPT	2025-12-31	96.8	97	95.2	95.8	3082600	-0.73
FPT	2026-01-05	95.7	95.7	93.2	95	7029300	-0.84
FPT	2026-01-06	94.7	94.8	93.6	94	5397500	-1.05
FPT	2026-01-07	94.1	97.6	94	97.5	11427400	3.72
FPT	2026-01-08	98.1	98.1	96.5	96.5	5576200	-1.03
FPT	2026-01-09	96.1	99.9	96	97.4	10048700	0.93
FPT	2026-01-12	98	99.5	98	99.5	9076600	2.16
FPT	2026-01-13	99.7	101.7	97.9	98.9	9411100	-0.6
FPT	2026-01-14	99	101.1	97.8	99.8	11900400	0.91
FPT	2026-01-15	99.8	101.7	98.4	98.5	7401400	-1.3
FPT	2026-01-16	99.5	105.3	99.5	105.3	23871100	6.9
FPT	2026-01-19	108.7	108.7	105.5	105.8	14710500	0.47
FPT	2026-01-20	106.1	107	104	104.1	12213100	-1.61
FPT	2026-01-21	103.2	106.5	101.1	105	14517300	0.86
FPT	2026-01-22	105.5	105.8	103.5	103.5	8095100	-1.43
FPT	2026-01-23	103.6	104.6	101	101	9418000	-2.42
FPT	2026-01-26	100.2	101.6	98	98.5	11183100	-2.48
FPT	2026-01-27	99.6	102.5	99.1	102.1	10585500	3.65
FPT	2026-01-28	102.4	105.8	102.3	104.5	11949800	2.35
FPT	2026-01-29	105	107.2	103.7	106.1	14247700	1.53
FPT	2026-01-30	106.4	106.4	103.9	104.5	10833200	-1.51
\.


--
-- Data for Name: frt; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.frt (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
FRT	2025-12-26	145	145	142	144.2	216500	0
FRT	2025-12-29	144.2	146	142.7	146	200200	1.25
FRT	2025-12-30	145.7	148.5	145	147.7	285100	1.16
FRT	2025-12-31	148.8	149.5	146.5	149.5	316800	1.22
FRT	2026-01-05	148.9	156.7	147.5	153.1	615800	2.41
FRT	2026-01-06	153	154.5	151.8	152.4	180000	-0.46
FRT	2026-01-07	152.6	156.9	152	152.6	389500	0.13
FRT	2026-01-08	152.6	152.6	145	146	883800	-4.33
FRT	2026-01-09	145.6	149.1	143.5	143.5	552500	-1.71
FRT	2026-01-12	143	146.4	142.9	145	312400	1.05
FRT	2026-01-13	144.6	146.6	144	144.6	461800	-0.28
FRT	2026-01-14	145.8	148.8	144.1	145	518200	0.28
FRT	2026-01-15	145	146	143.5	146	350700	0.69
FRT	2026-01-16	146	156.2	144.5	155	1373100	6.16
FRT	2026-01-19	155.9	155.9	150.6	153.4	295600	-1.03
FRT	2026-01-20	153	160	152.3	152.8	698300	-0.39
FRT	2026-01-21	151.5	152.8	148	152.8	496000	0
FRT	2026-01-22	152.8	153.4	149.6	153.3	442300	0.33
FRT	2026-01-23	152.9	155.2	151.1	151.8	389500	-0.98
FRT	2026-01-26	151.1	151.7	145.1	145.1	485600	-4.41
FRT	2026-01-27	145.2	155.1	145.2	154.7	791600	6.62
FRT	2026-01-28	155.6	162.1	155	160.3	1500000	3.62
FRT	2026-01-29	160	168.9	159	165	1082600	2.93
FRT	2026-01-30	168.9	171	165	169.9	998500	2.97
\.


--
-- Data for Name: fts; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.fts (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
FTS	2025-12-26	33.6	34.85	33.15	34.35	3033200	0
FTS	2025-12-29	34.5	34.75	33.8	34.3	1172300	-0.15
FTS	2025-12-30	34.65	34.65	33.6	33.6	1024400	-2.04
FTS	2025-12-31	33.6	33.85	32.95	32.95	1777900	-1.93
FTS	2026-01-05	32.95	33.2	31.5	31.8	1743900	-3.49
FTS	2026-01-06	31.8	32.55	31.3	32.2	1343600	1.26
FTS	2026-01-07	32.25	32.95	32.05	32.65	996400	1.4
FTS	2026-01-08	32.8	33.3	32.05	32.7	1366700	0.15
FTS	2026-01-09	33.3	33.35	32.2	32.3	1168900	-1.22
FTS	2026-01-12	32.5	34.55	32.45	34.55	3833700	6.97
FTS	2026-01-13	34.65	35.05	33.9	34	2260200	-1.59
FTS	2026-01-14	34	34.4	33.4	34	2273700	0
FTS	2026-01-15	33.4	34.25	33.4	33.65	1483800	-1.03
FTS	2026-01-16	33.7	35.4	33.5	34.25	3423500	1.78
FTS	2026-01-19	34.5	34.6	33.9	33.9	1233600	-1.02
FTS	2026-01-20	34.2	34.2	33.25	33.25	1898900	-1.92
FTS	2026-01-21	33.25	33.3	32.1	32.35	2350500	-2.71
FTS	2026-01-22	32.95	33.1	32.35	32.85	1157600	1.55
FTS	2026-01-23	32.85	33.6	32.5	32.6	873300	-0.76
FTS	2026-01-26	32.45	32.9	31.95	32.25	1016800	-1.07
FTS	2026-01-27	31.95	32.5	31.95	32.25	1205900	0
FTS	2026-01-28	32.25	32.75	32.1	32.45	922500	0.62
FTS	2026-01-29	32.35	32.8	32.35	32.6	420200	0.46
FTS	2026-01-30	32.6	33.05	32.4	32.9	1120300	0.92
\.


--
-- Data for Name: gas; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.gas (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
GAS	2025-12-26	69	70.9	68	70.5	2428400	0
GAS	2025-12-29	70.6	75.4	70.2	75.1	5613700	6.52
GAS	2025-12-30	75.1	75.7	73.8	74.9	1764400	-0.27
GAS	2025-12-31	74.5	75.8	72.4	72.4	1872800	-3.34
GAS	2026-01-05	73.1	77.4	72.8	77.4	5364200	6.91
GAS	2026-01-06	77.8	82.8	77.8	82.8	4730300	6.98
GAS	2026-01-07	85.2	88.5	85.2	88.5	4173200	6.88
GAS	2026-01-08	93.5	94.6	90	91.6	6171100	3.5
GAS	2026-01-09	90.6	98	89.8	97.1	4592600	6
GAS	2026-01-12	101	102.5	94.5	97	5514400	-0.1
GAS	2026-01-13	97	103.7	96	103.7	4960100	6.91
GAS	2026-01-14	107	110.9	100	107	4910800	3.18
GAS	2026-01-15	106.9	107.1	102.3	103	3558000	-3.74
GAS	2026-01-16	103.2	107.2	99.8	99.8	4537900	-3.11
GAS	2026-01-19	100	106.2	98.1	105.8	4498200	6.01
GAS	2026-01-20	106	110.8	104	104.5	3888800	-1.23
GAS	2026-01-21	101.2	110	101	110	4569400	5.26
GAS	2026-01-22	110.5	110.5	104.7	104.8	4035100	-4.73
GAS	2026-01-23	104.8	106.3	99.2	100.8	3428600	-3.82
GAS	2026-01-26	100.5	107.8	100.5	107.8	6102800	6.94
GAS	2026-01-27	109.8	115.3	109	115.3	5889900	6.96
GAS	2026-01-28	121	122.8	115	118	7140900	2.34
GAS	2026-01-29	118.5	118.5	111.3	116.8	3705700	-1.02
GAS	2026-01-30	116.8	122	115.8	117	3644900	0.17
\.


--
-- Data for Name: gee; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.gee (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
GEE	2025-12-26	202.3	216	197.2	216	1327800	0
GEE	2025-12-29	217	230.8	216	230.8	1422400	6.85
GEE	2025-12-30	231.7	241.3	231.7	241	825600	4.42
GEE	2025-12-31	241.8	245.8	238.8	245.5	1079800	1.87
GEE	2026-01-05	245.8	246	228.4	229	832500	-6.72
GEE	2026-01-06	229.5	232	224	231	1035300	0.87
GEE	2026-01-07	231.5	231.5	220	226.5	660800	-1.95
GEE	2026-01-08	226.7	226.7	211.1	212	527200	-6.4
GEE	2026-01-09	212.8	213	200	205	387300	-3.3
GEE	2026-01-12	205.5	210	198.8	209.5	319100	2.2
GEE	2026-01-13	210	224.1	208.1	218	1136300	4.06
GEE	2026-01-14	218.5	218.8	208.5	209	309500	-4.13
GEE	2026-01-15	209.6	223.6	208	223.6	1465700	6.99
GEE	2026-01-16	226.1	239.2	223.9	229	1260700	2.42
GEE	2026-01-19	229.6	235.3	218	229	813200	0
GEE	2026-01-20	229.2	229.6	216	216	492600	-5.68
GEE	2026-01-21	216	216	201.1	201.3	943500	-6.81
GEE	2026-01-22	202.5	209.8	202	204	437100	1.34
GEE	2026-01-23	204.8	206.3	198.4	203	459000	-0.49
GEE	2026-01-26	203.5	203.5	190	191	246000	-5.91
GEE	2026-01-27	192.1	196.6	191	195.1	335400	2.15
GEE	2026-01-28	196	199	181.8	184.5	660200	-5.43
GEE	2026-01-29	185.6	186	178	186	236200	0.81
GEE	2026-01-30	187	187	179.2	180.8	210200	-2.8
\.


--
-- Data for Name: gex; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.gex (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
GEX	2025-12-26	42.7	42.95	40.6	42.75	10069500	0
GEX	2025-12-29	43	44.3	42.9	44	6002700	2.92
GEX	2025-12-30	44.3	44.8	43.6	44.2	6168100	0.45
GEX	2025-12-31	44.45	44.45	43.6	43.6	4658800	-1.36
GEX	2026-01-05	43.75	43.75	40.55	41	12340400	-5.96
GEX	2026-01-06	41.2	42.6	40.75	42	7562200	2.44
GEX	2026-01-07	42.5	42.75	41.7	42.25	6965700	0.6
GEX	2026-01-08	42.35	42.9	40.95	42	9588500	-0.59
GEX	2026-01-09	42.1	42.1	39.15	39.25	15147400	-6.55
GEX	2026-01-12	39.35	41.4	38.05	40.5	12530100	3.18
GEX	2026-01-13	41	43.3	40.25	43	25539000	6.17
GEX	2026-01-14	43	43.4	41.2	41.5	13996300	-3.49
GEX	2026-01-15	41.85	42.35	40.75	42.2	11444800	1.69
GEX	2026-01-16	42.7	42.7	40.8	40.8	10907600	-3.32
GEX	2026-01-19	41	42.65	40.75	41.5	9472600	1.72
GEX	2026-01-20	41.85	42.4	40.7	40.95	9663300	-1.33
GEX	2026-01-21	41	41	39.05	39.2	12481600	-4.27
GEX	2026-01-22	39.8	40.45	39.5	39.95	6267000	1.91
GEX	2026-01-23	40.2	40.45	39.1	39.2	6772100	-1.88
GEX	2026-01-26	39.3	39.65	36.5	36.7	16567500	-6.38
GEX	2026-01-27	37.2	37.2	36.1	36.7	8486300	0
GEX	2026-01-28	37	37.2	35.1	36	9439800	-1.91
GEX	2026-01-29	36.5	37.4	36	36.5	4464400	1.39
GEX	2026-01-30	37.1	37.3	36.25	36.7	7438900	0.55
\.


--
-- Data for Name: gmd; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.gmd (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
GMD	2025-12-26	60	60.2	58.5	59.5	819700	0
GMD	2025-12-29	59.4	61.3	59.4	61.3	1398500	3.03
GMD	2025-12-30	61	62	61	61.8	1094900	0.82
GMD	2025-12-31	62	62	61	61	318600	-1.29
GMD	2026-01-05	61	61.2	60.3	60.6	636500	-0.66
GMD	2026-01-06	60.6	61.2	60.3	60.4	711700	-0.33
GMD	2026-01-07	60.3	63.1	60.2	62.7	2595000	3.81
GMD	2026-01-08	62.7	63.3	61.9	63	1995300	0.48
GMD	2026-01-09	63	63.2	62	62	820800	-1.59
GMD	2026-01-12	62.4	63.6	62.4	63	1240600	1.61
GMD	2026-01-13	62.8	64	62.6	62.9	1020900	-0.16
GMD	2026-01-14	63.2	63.8	62.6	63	1537600	0.16
GMD	2026-01-15	63.2	63.9	63	63.5	933500	0.79
GMD	2026-01-16	63.5	66	63.1	63.9	1894500	0.63
GMD	2026-01-19	64.4	64.4	63.6	63.8	966900	-0.16
GMD	2026-01-20	64	68.2	63.7	68.2	4470900	6.9
GMD	2026-01-21	68.5	71.2	68.5	70.5	5343400	3.37
GMD	2026-01-22	70.6	70.6	68.7	70	2686400	-0.71
GMD	2026-01-23	70.1	70.1	66.6	68.5	2548100	-2.14
GMD	2026-01-26	67.8	68.2	64	64	3116600	-6.57
GMD	2026-01-27	63.3	65.8	63.3	65.8	1214600	2.81
GMD	2026-01-28	65.6	66.1	64.9	65.8	825100	0
GMD	2026-01-29	65.2	67.9	65.2	67.9	995300	3.19
GMD	2026-01-30	68	70	67	68.5	1901300	0.88
\.


--
-- Data for Name: gvr; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.gvr (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
GVR	2025-12-26	25.55	25.95	25.2	25.5	1406500	0
GVR	2025-12-29	25.6	26.2	25.6	26	1106300	1.96
GVR	2025-12-30	26.2	26.9	26.15	26.6	2111300	2.31
GVR	2025-12-31	26.6	26.8	26.2	26.2	1021300	-1.5
GVR	2026-01-05	26.2	26.45	25.5	25.75	1408600	-1.72
GVR	2026-01-06	25.85	27.55	25.85	27.55	4061800	6.99
GVR	2026-01-07	28.55	29.45	28.5	29.45	13681500	6.9
GVR	2026-01-08	30.9	30.9	29.6	30.05	10234200	2.04
GVR	2026-01-09	30.4	32	30.3	31.6	9394000	5.16
GVR	2026-01-12	32.3	32.95	31	32.2	8946600	1.9
GVR	2026-01-13	32.1	34.45	31.55	34	14389700	5.59
GVR	2026-01-14	34.3	36.35	33.7	36.35	14095400	6.91
GVR	2026-01-15	36.35	38.85	35.8	37.35	11037100	2.75
GVR	2026-01-16	37.4	38.5	35.55	35.9	8596200	-3.88
GVR	2026-01-19	36	38.4	35.95	38	9356800	5.85
GVR	2026-01-20	39.2	40.65	37.6	39	10142500	2.63
GVR	2026-01-21	38	39.5	36.5	38.5	8356100	-1.28
GVR	2026-01-22	39.05	40.8	38.5	39	7081600	1.3
GVR	2026-01-23	38.55	38.85	36.6	36.9	6548400	-5.38
GVR	2026-01-26	36.65	39.35	36.65	38.65	8681600	4.74
GVR	2026-01-27	38.7	40.6	38.4	39.85	8227600	3.1
GVR	2026-01-28	41.15	42.4	38.2	38.2	13817900	-4.14
GVR	2026-01-29	38.2	40.85	37.9	40.85	9435700	6.94
GVR	2026-01-30	40.85	41.8	39.65	39.85	6315900	-2.45
\.


--
-- Data for Name: hag; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hag (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
HAG	2025-12-26	18.2	18.4	17.35	17.9	22594600	0
HAG	2025-12-29	17.8	17.85	17.45	17.6	7249800	-1.68
HAG	2025-12-30	17.65	17.9	17.6	17.75	7172000	0.85
HAG	2025-12-31	17.8	17.9	17.55	17.55	5528700	-1.13
HAG	2026-01-05	17.55	17.75	16.65	16.85	14563700	-3.99
HAG	2026-01-06	16.95	17.55	16.8	16.9	12015200	0.3
HAG	2026-01-07	17.05	17.2	16.8	17.05	9665800	0.89
HAG	2026-01-08	17.2	17.45	16.95	17.2	9700100	0.88
HAG	2026-01-09	17.15	17.35	16.35	16.6	15694100	-3.49
HAG	2026-01-12	16.5	17.1	16.4	17.1	8352500	3.01
HAG	2026-01-13	17.1	17.5	16.85	17.3	9918600	1.17
HAG	2026-01-14	17.4	17.45	17.05	17.35	9407800	0.29
HAG	2026-01-15	17.25	17.85	17.1	17.75	11058500	2.31
HAG	2026-01-16	17.75	17.9	17.4	17.75	8520300	0
HAG	2026-01-19	18	18.4	17.55	17.55	15255900	-1.13
HAG	2026-01-20	17.6	17.95	17.55	17.95	9454500	2.28
HAG	2026-01-21	17.8	18.4	17.55	17.85	13414700	-0.56
HAG	2026-01-22	18	18.2	17.75	17.85	8153400	0
HAG	2026-01-23	17.9	18	17.55	17.95	9447500	0.56
HAG	2026-01-26	17.95	17.95	16.7	16.7	20665400	-6.96
HAG	2026-01-27	16.8	16.9	16.6	16.7	5676700	0
HAG	2026-01-28	16.95	16.95	16.5	16.9	6725100	1.2
HAG	2026-01-29	16.9	17.1	16.75	17.1	6742000	1.18
HAG	2026-01-30	17.1	17.25	16.95	17.25	3821300	0.88
\.


--
-- Data for Name: hcm; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hcm (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
HCM	2025-12-26	22.42	22.72	21.98	22.52	12140069	0
HCM	2025-12-29	22.62	22.62	22.32	22.37	4179959	-0.67
HCM	2025-12-30	22.42	22.52	22.22	22.32	3421259	-0.22
HCM	2025-12-31	22.32	22.42	22.03	22.03	5474998	-1.3
HCM	2026-01-05	22.08	22.18	20.95	21.39	9136531	-2.91
HCM	2026-01-06	21.44	21.63	20.95	21.39	7294524	0
HCM	2026-01-07	21.54	21.98	21.49	21.88	6003252	2.29
HCM	2026-01-08	21.93	22.62	21.93	22.13	11689002	1.14
HCM	2026-01-09	22.27	22.47	22.03	22.22	9072990	0.41
HCM	2026-01-12	22.86	23.75	22.81	23.75	23802689	6.89
HCM	2026-01-13	25.27	25.37	24.83	25.08	53798824	5.6
HCM	2026-01-14	25.18	26.06	25.08	26.06	39632257	3.91
HCM	2026-01-15	26.06	26.6	25.62	25.62	32285695	-1.69
HCM	2026-01-16	25.57	27.04	25.18	26.06	32574901	1.72
HCM	2026-01-19	26.11	26.16	25.42	25.47	22984730	-2.26
HCM	2026-01-20	25.67	26.01	25.08	25.08	24250571	-1.53
HCM	2026-01-21	23.9	23.99	23.36	23.36	78441356	-6.86
HCM	2026-01-22	23.6	23.75	23.16	23.26	17619989	-0.43
HCM	2026-01-23	23.5	23.95	23.16	23.26	16211189	0
HCM	2026-01-26	23.11	23.55	22.27	22.52	18578385	-3.18
HCM	2026-01-27	22.57	23.36	22.42	22.96	13059972	1.95
HCM	2026-01-28	23.01	23.21	22.77	23.06	9160347	0.44
HCM	2026-01-29	23.06	23.5	23.06	23.21	8063468	0.65
HCM	2026-01-30	23.31	23.4	23.11	23.16	10288031	-0.22
\.


--
-- Data for Name: hdb; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hdb (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
HDB	2025-12-26	27	28	26.8	27.6	21304775	0
HDB	2025-12-29	27.7	27.7	27.15	27.6	14642738	0
HDB	2025-12-30	27.6	27.95	27	27.95	29044307	1.27
HDB	2025-12-31	27.8	29.7	27.75	29.7	43989500	6.26
HDB	2026-01-05	29.5	29.5	28	28.95	24967900	-2.53
HDB	2026-01-06	28.8	29.05	28.1	29.05	14635600	0.35
HDB	2026-01-07	29.05	29.55	28.35	29.1	23080000	0.17
HDB	2026-01-08	29.15	29.2	28.5	28.8	17601200	-1.03
HDB	2026-01-09	28.85	29.1	27.5	27.9	29852300	-3.13
HDB	2026-01-12	27.7	28.35	27.25	28.2	23631100	1.08
HDB	2026-01-13	28.2	28.45	28	28.45	16194800	0.89
HDB	2026-01-14	28.5	28.5	27.55	27.65	21961000	-2.81
HDB	2026-01-15	27.5	29.55	26.8	29.55	32989100	6.87
HDB	2026-01-16	29.5	29.5	28.45	28.45	15902500	-3.72
HDB	2026-01-19	28.5	29	28.2	28.75	16574300	1.05
HDB	2026-01-20	29	29.8	28.75	28.95	19555100	0.7
HDB	2026-01-21	28.65	30	28.3	29.05	24503900	0.35
HDB	2026-01-22	29.1	29.75	29.05	29.2	18695900	0.52
HDB	2026-01-23	29.25	29.6	28.65	29.6	16077900	1.37
HDB	2026-01-26	29.4	29.4	28.3	28.35	11703400	-4.22
HDB	2026-01-27	28.35	29	27.8	29	15743500	2.29
HDB	2026-01-28	28.5	28.85	28.35	28.5	24360600	-1.72
HDB	2026-01-29	28.55	28.6	27.8	27.85	13587000	-2.28
HDB	2026-01-30	27.9	28.4	27.8	28.3	15990600	1.62
\.


--
-- Data for Name: hdc; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hdc (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
HDC	2025-12-26	23.3	23.65	22.4	22.8	4164500	0
HDC	2025-12-29	22.8	23.35	22.8	23.1	1932700	1.32
HDC	2025-12-30	23.15	23.3	22.7	22.7	1881500	-1.73
HDC	2025-12-31	22.85	23.1	22.7	22.7	2288700	0
HDC	2026-01-05	22.85	23.05	21.9	22.25	2837800	-1.98
HDC	2026-01-06	22.25	22.45	21.5	21.75	3477700	-2.25
HDC	2026-01-07	22	22.2	21.6	22	2420200	1.15
HDC	2026-01-08	22.1	22.6	21.7	21.8	2997800	-0.91
HDC	2026-01-09	21.8	21.8	20.5	20.65	6737000	-5.28
HDC	2026-01-12	20.65	22	20.1	21.7	5006600	5.08
HDC	2026-01-13	22.15	22.3	21.8	21.9	4214500	0.92
HDC	2026-01-14	21.6	22.05	21.05	21.1	6277800	-3.65
HDC	2026-01-15	21.1	21.9	21.1	21.55	3989900	2.13
HDC	2026-01-16	21.9	21.9	21.2	21.3	3448600	-1.16
HDC	2026-01-19	21.35	21.7	21.2	21.35	2313600	0.23
HDC	2026-01-20	21.6	21.65	21.15	21.15	3042600	-0.94
HDC	2026-01-21	21.25	21.8	21.05	21.35	4538100	0.95
HDC	2026-01-22	21.6	22.8	21.3	22.8	5372000	6.79
HDC	2026-01-23	23	23	21.8	21.8	5809700	-4.39
HDC	2026-01-26	21.9	21.95	20.8	20.95	4253300	-3.9
HDC	2026-01-27	21	21.25	20.85	21	2462900	0.24
HDC	2026-01-28	21	21.95	20.5	21.5	4901400	2.38
HDC	2026-01-29	21.75	22.1	21.35	21.35	2543100	-0.7
HDC	2026-01-30	21.6	22.15	21.5	21.7	4243100	1.64
\.


--
-- Data for Name: hdg; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hdg (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
HDG	2025-12-26	27.45	27.5	26.51	27.1	1902955	0
HDG	2025-12-29	27.3	27.3	26.51	27.25	1773676	0.55
HDG	2025-12-30	26.91	27.2	26.56	27.1	2216848	-0.55
HDG	2025-12-31	27.1	27.1	26.76	26.81	1158916	-1.07
HDG	2026-01-05	26.86	27.05	25.53	25.63	3300973	-4.4
HDG	2026-01-06	25.83	26.02	25.19	25.53	3641041	-0.39
HDG	2026-01-07	25.63	26.46	25.63	26.32	2828026	3.09
HDG	2026-01-08	26.61	26.61	25.83	25.83	2947243	-1.86
HDG	2026-01-09	25.92	26.42	25.53	25.83	5294207	0
HDG	2026-01-12	25.58	26.61	25.24	26.37	3950223	2.09
HDG	2026-01-13	26.96	26.96	26.12	26.51	2931296	0.53
HDG	2026-01-14	26.51	26.86	25.83	26.02	3997133	-1.85
HDG	2026-01-15	26.02	26.51	25.73	26.12	2808660	0.38
HDG	2026-01-16	26.17	26.37	25.68	25.78	2276608	-1.3
HDG	2026-01-19	26.02	26.12	25.78	25.78	1845541	0
HDG	2026-01-20	25.88	26.17	25.73	25.97	2186820	0.74
HDG	2026-01-21	25.92	25.97	25.34	25.43	2670411	-2.08
HDG	2026-01-22	25.48	26.91	25.48	26.46	3596420	4.05
HDG	2026-01-23	26.46	26.76	25.83	25.83	1598545	-2.38
HDG	2026-01-26	25.83	25.83	24.6	24.75	2730900	-4.18
HDG	2026-01-27	24.75	25.58	24.5	25.29	2309240	2.18
HDG	2026-01-28	24.94	25.83	24.94	25.43	1877683	0.55
HDG	2026-01-29	25.83	25.92	25.58	25.92	1090646	1.93
HDG	2026-01-30	26.07	27.2	26.07	26.96	5365124	4.01
\.


--
-- Data for Name: hhv; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hhv (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
HHV	2025-12-26	13.2	13.35	12.95	13.05	4790108	0
HHV	2025-12-29	13.05	13.2	12.95	13.1	3131875	0.38
HHV	2025-12-30	13.15	13.25	13.05	13.1	5106571	0
HHV	2025-12-31	13.2	13.25	13.1	13.15	2197654	0.38
HHV	2026-01-05	13.15	13.25	12.5	12.5	5913454	-4.94
HHV	2026-01-06	12.7	12.75	12.2	12.5	6124584	0
HHV	2026-01-07	12.55	12.8	12.5	12.75	4151004	2
HHV	2026-01-08	12.85	12.95	12.6	12.65	4175802	-0.78
HHV	2026-01-09	12.8	12.8	12.35	12.4	5164400	-1.98
HHV	2026-01-12	12.2	12.95	12.2	12.9	7841500	4.03
HHV	2026-01-13	13.1	13.15	12.85	12.85	4871800	-0.39
HHV	2026-01-14	12.85	13.2	12.8	12.9	7501800	0.39
HHV	2026-01-15	12.9	13.1	12.75	12.85	5491100	-0.39
HHV	2026-01-16	12.9	13.1	12.8	12.8	4838800	-0.39
HHV	2026-01-19	12.85	13.3	12.85	12.95	6449400	1.17
HHV	2026-01-20	13.15	13.15	12.9	12.9	4723300	-0.39
HHV	2026-01-21	12.95	12.95	12.6	12.65	4896700	-1.94
HHV	2026-01-22	12.75	13	12.6	12.9	5568800	1.98
HHV	2026-01-23	12.95	13	12.65	12.7	2494000	-1.55
HHV	2026-01-26	12.7	12.75	12.15	12.3	6194100	-3.15
HHV	2026-01-27	12.25	12.3	12	12.1	4519500	-1.63
HHV	2026-01-28	12.15	12.3	12.05	12.2	2515500	0.83
HHV	2026-01-29	12.25	12.3	12.1	12.2	2463900	0
HHV	2026-01-30	12.2	12.3	12.15	12.2	4458200	0
\.


--
-- Data for Name: hpg; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hpg (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
HPG	2025-12-26	27.1	27.7	26.65	26.9	94035000	0
HPG	2025-12-29	27	27	26.65	26.7	16525200	-0.74
HPG	2025-12-30	26.7	26.85	26.5	26.5	24581900	-0.75
HPG	2025-12-31	26.45	26.6	26.35	26.4	20149700	-0.38
HPG	2026-01-05	26.4	26.55	25.7	25.95	46487900	-1.7
HPG	2026-01-06	26	26.2	25.35	25.95	39616600	0
HPG	2026-01-07	25.95	26.75	25.95	26.6	37249700	2.5
HPG	2026-01-08	26.7	26.8	26.4	26.4	37518600	-0.75
HPG	2026-01-09	26.8	26.8	26.2	26.2	35089300	-0.76
HPG	2026-01-12	26.35	27.6	26.35	27.5	83534900	4.96
HPG	2026-01-13	27.75	27.9	27	27.25	45777400	-0.91
HPG	2026-01-14	27.45	27.6	27	27.25	53908300	0
HPG	2026-01-15	27.25	27.9	27.25	27.6	61937700	1.28
HPG	2026-01-16	27.8	28.25	27.3	27.6	65577000	0
HPG	2026-01-19	27.9	28.2	27.65	27.7	40465900	0.36
HPG	2026-01-20	27.85	27.95	27.1	27.25	44169600	-1.62
HPG	2026-01-21	27.15	27.25	26.65	26.8	37234300	-1.65
HPG	2026-01-22	26.9	27.3	26.85	26.85	27732600	0.19
HPG	2026-01-23	27.1	27.15	26.75	26.75	24233600	-0.37
HPG	2026-01-26	26.75	26.8	26.15	26.3	33759300	-1.68
HPG	2026-01-27	26.35	26.55	26.2	26.45	21014200	0.57
HPG	2026-01-28	26.55	27.1	26.55	26.7	29982700	0.95
HPG	2026-01-29	26.8	27.1	26.8	27.1	27608700	1.5
HPG	2026-01-30	27.3	27.5	26.75	26.8	40558400	-1.11
\.


--
-- Data for Name: hsg; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.hsg (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
HSG	2025-12-26	16.45	16.7	16	16.05	4457400	0
HSG	2025-12-29	16.3	16.3	15.9	16.2	2264900	0.93
HSG	2025-12-30	16.2	16.2	15.9	15.9	1212100	-1.85
HSG	2025-12-31	15.95	16	15.7	15.75	1937700	-0.94
HSG	2026-01-05	15.75	15.85	15.4	15.45	2425200	-1.9
HSG	2026-01-06	15.45	15.55	15.15	15.4	2138900	-0.32
HSG	2026-01-07	15.5	15.9	15.45	15.85	2595400	2.92
HSG	2026-01-08	15.95	16	15.75	15.75	2664100	-0.63
HSG	2026-01-09	15.8	15.9	15.6	15.6	2336100	-0.95
HSG	2026-01-12	15.6	16.4	15.6	16.3	5037000	4.49
HSG	2026-01-13	16.55	16.6	16.15	16.2	4486500	-0.61
HSG	2026-01-14	16.15	16.45	16.1	16.3	5272200	0.62
HSG	2026-01-15	16.4	17.1	16.3	17	12792100	4.29
HSG	2026-01-16	17.25	17.25	16.65	16.7	7176700	-1.76
HSG	2026-01-19	16.85	16.95	16.6	16.8	3059000	0.6
HSG	2026-01-20	16.8	16.9	16.6	16.75	3250400	-0.3
HSG	2026-01-21	16.7	16.75	16.15	16.3	5402500	-2.69
HSG	2026-01-22	16.35	16.7	16.3	16.65	3560100	2.15
HSG	2026-01-23	16.65	16.8	16.35	16.35	2336500	-1.8
HSG	2026-01-26	16.35	16.35	15.75	15.9	5079600	-2.75
HSG	2026-01-27	15.7	15.95	15.6	15.9	3184200	0
HSG	2026-01-28	15.95	16.2	15.9	16.2	2676600	1.89
HSG	2026-01-29	16.3	16.4	16.15	16.25	2564700	0.31
HSG	2026-01-30	16.35	16.5	16.1	16.1	1957400	-0.92
\.


--
-- Data for Name: ht1; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ht1 (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
HT1	2025-12-26	15.3	15.5	15	15.45	162800	0
HT1	2025-12-29	15.45	15.5	15.1	15.3	97100	-0.97
HT1	2025-12-30	15.05	15.6	14.9	15.6	270700	1.96
HT1	2025-12-31	15.5	16.5	15.2	16.5	349700	5.77
HT1	2026-01-05	16	16	15.5	15.5	270300	-6.06
HT1	2026-01-06	15.5	15.75	15.25	15.6	154300	0.65
HT1	2026-01-07	15.4	15.9	15.25	15.7	224900	0.64
HT1	2026-01-08	16	16.2	15.75	15.8	514100	0.64
HT1	2026-01-09	15.85	16.9	15.85	16.5	1314400	4.43
HT1	2026-01-12	16.55	16.9	16.3	16.5	468700	0
HT1	2026-01-13	16.6	16.65	16.2	16.3	535400	-1.21
HT1	2026-01-14	16.35	17.1	16	16.1	988700	-1.23
HT1	2026-01-15	16.2	16.8	16.2	16.7	634900	3.73
HT1	2026-01-16	17.25	17.25	16.8	16.85	723500	0.9
HT1	2026-01-19	16.9	17.3	16.9	17.05	485500	1.19
HT1	2026-01-20	17.2	17.35	16.8	16.8	506100	-1.47
HT1	2026-01-21	16.7	16.9	16.15	16.3	1097300	-2.98
HT1	2026-01-22	16.6	16.9	16.3	16.3	900900	0
HT1	2026-01-23	16.3	16.7	16.3	16.45	730300	0.92
HT1	2026-01-26	16.35	16.7	15.5	15.5	1307900	-5.78
HT1	2026-01-27	15.5	15.75	14.75	15	1105200	-3.23
HT1	2026-01-28	15	15.45	14.7	14.85	778900	-1
HT1	2026-01-29	14.85	15.25	14.65	14.85	818000	0
HT1	2026-01-30	15.15	15.15	14.75	14.85	825700	0
\.


--
-- Data for Name: imp; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.imp (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
IMP	2025-12-26	50.6	50.6	49.2	50.5	71800	0
IMP	2025-12-29	50.4	50.7	50	50.7	34000	0.4
IMP	2025-12-30	50.7	52.4	50.7	51.7	103500	1.97
IMP	2025-12-31	51.7	53.2	51.7	52.5	96300	1.55
IMP	2026-01-05	52.5	52.5	51.6	52.1	69500	-0.76
IMP	2026-01-06	52.1	52.3	50.5	51.8	31400	-0.58
IMP	2026-01-07	51.3	52.5	51.2	52.5	90300	1.35
IMP	2026-01-08	52.4	52.8	52	52.5	36500	0
IMP	2026-01-09	52.3	52.6	52	52.5	43800	0
IMP	2026-01-12	52.3	52.7	52.2	52.3	61400	-0.38
IMP	2026-01-13	52.1	53.6	52.1	53.1	88100	1.53
IMP	2026-01-14	53.1	53.6	53.1	53.2	47900	0.19
IMP	2026-01-15	53.2	53.7	53	53.7	57700	0.94
IMP	2026-01-16	54.3	54.6	54	54.2	56600	0.93
IMP	2026-01-19	54.5	55	54.4	54.5	28100	0.55
IMP	2026-01-20	54.9	54.9	53.6	53.6	61100	-1.65
IMP	2026-01-21	53.6	53.9	53.6	53.6	83800	0
IMP	2026-01-22	54.4	54.4	53.5	53.6	31000	0
IMP	2026-01-23	53.4	54	53.4	53.7	86600	0.19
IMP	2026-01-26	53.6	54	53.6	53.9	27500	0.37
IMP	2026-01-27	53	53.7	53	53.7	23200	-0.37
IMP	2026-01-28	53.7	53.8	53.6	53.8	31600	0.19
IMP	2026-01-29	53.6	53.8	53.6	53.8	47900	0
IMP	2026-01-30	53.8	54.3	53.8	54.2	51500	0.74
\.


--
-- Data for Name: kbc; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.kbc (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
KBC	2025-12-26	33.45	33.55	32.45	33.2	2665800	0
KBC	2025-12-29	33.2	33.75	33	33.6	1493800	1.2
KBC	2025-12-30	33.65	34.6	33.65	34.4	3208500	2.38
KBC	2025-12-31	34.65	35.9	34.55	35.35	7407300	2.76
KBC	2026-01-05	36	36	34.4	34.7	2426000	-1.84
KBC	2026-01-06	34.7	35.75	34.7	34.95	3389100	0.72
KBC	2026-01-07	35.3	35.8	34.95	35.8	4480500	2.43
KBC	2026-01-08	36.1	36.25	35	35.6	6488400	-0.56
KBC	2026-01-09	35.65	37.3	34.6	36	9412500	1.12
KBC	2026-01-12	36.1	36.75	35.05	36.2	4963200	0.56
KBC	2026-01-13	36.4	37.35	35.65	35.85	6826700	-0.97
KBC	2026-01-14	35.8	36.7	35.4	36.05	8609500	0.56
KBC	2026-01-15	36	36.8	35.6	36.3	6572000	0.69
KBC	2026-01-16	36.4	36.6	35.6	35.6	4829800	-1.93
KBC	2026-01-19	35.7	38.05	35.5	38	20476900	6.74
KBC	2026-01-20	38.1	38.1	37.25	37.65	7541100	-0.92
KBC	2026-01-21	37.3	37.95	36.05	37	9057800	-1.73
KBC	2026-01-22	37.1	37.45	36.5	37.4	5341400	1.08
KBC	2026-01-23	37.4	37.45	35.85	36.3	5310000	-2.94
KBC	2026-01-26	36.2	36.3	33.8	33.8	11290300	-6.89
KBC	2026-01-27	33.5	34.15	33.3	33.85	5410200	0.15
KBC	2026-01-28	33.9	34.65	33.8	34.1	4138100	0.74
KBC	2026-01-29	34.1	34.4	33.5	33.6	2989200	-1.47
KBC	2026-01-30	34	35	33.8	34.9	4796300	3.87
\.


--
-- Data for Name: kdc; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.kdc (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
KDC	2025-12-26	50.1	50.78	49.81	50.78	217926	0
KDC	2025-12-29	50.78	50.78	50.1	50.69	273663	-0.18
KDC	2025-12-30	50.39	50.78	50.29	50.78	252019	0.18
KDC	2025-12-31	50.29	50.78	50.2	50.78	277554	0
KDC	2026-01-05	50.78	50.78	50.2	50.78	243069	0
KDC	2026-01-06	50.59	50.59	49.81	50.2	282346	-1.14
KDC	2026-01-07	50.29	50.59	46.73	48.44	315604	-3.51
KDC	2026-01-08	49.42	50.49	48.83	50.49	347038	4.23
KDC	2026-01-09	50.49	50.49	49.22	50.29	286526	-0.4
KDC	2026-01-12	50.29	50.59	49.61	50.1	283913	-0.38
KDC	2026-01-13	50.2	50.4	49.65	50	347363	-0.2
KDC	2026-01-14	49.8	50.4	49.6	50.4	261275	0.8
KDC	2026-01-15	50.3	50.3	49.7	50.3	279287	-0.2
KDC	2026-01-16	50.1	50.3	49.7	50.3	210855	0
KDC	2026-01-19	50.2	50.2	49.75	50	204182	-0.6
KDC	2026-01-20	49.5	50.2	49.5	50.2	279845	0.4
KDC	2026-01-21	49.65	50.4	49.55	50.4	214907	0.4
KDC	2026-01-22	49.8	50.2	49.8	50.2	200392	-0.4
KDC	2026-01-23	50.2	50.2	49.8	50.1	197599	-0.2
KDC	2026-01-26	49.95	50	49.5	50	185100	-0.2
KDC	2026-01-27	50	50	49.5	50	161700	0
KDC	2026-01-28	49.9	50	49.5	50	221600	0
KDC	2026-01-29	49.95	50	49.6	50	193600	0
KDC	2026-01-30	49.95	50	49.65	50	188600	0
\.


--
-- Data for Name: kdh; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.kdh (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
KDH	2025-12-26	31.7	32.05	30.65	32.05	4535100	0
KDH	2025-12-29	32	32.05	31.7	31.7	1693100	-1.09
KDH	2025-12-30	31.8	31.85	31.2	31.5	2361200	-0.63
KDH	2025-12-31	31.5	31.6	31.1	31.5	2300200	0
KDH	2026-01-05	31.5	32.5	30.85	31.9	6288800	1.27
KDH	2026-01-06	32.05	32.05	31.45	31.85	4376900	-0.16
KDH	2026-01-07	31.85	31.85	31.05	31.35	5218500	-1.57
KDH	2026-01-08	31.4	31.6	30.2	30.5	8390100	-2.71
KDH	2026-01-09	30.5	30.6	28.5	28.6	14633500	-6.23
KDH	2026-01-12	28.7	29.85	28.25	29.2	6489400	2.1
KDH	2026-01-13	30	30.35	29.9	30	5735900	2.74
KDH	2026-01-14	29.95	30.05	28.75	29.05	8278700	-3.17
KDH	2026-01-15	28.75	30.3	28.65	30.1	9028300	3.61
KDH	2026-01-16	30.4	30.85	29.65	29.75	6436700	-1.16
KDH	2026-01-19	29.9	29.95	29.3	29.3	5483700	-1.51
KDH	2026-01-20	29.3	29.65	28.95	29.3	5270000	0
KDH	2026-01-21	29.2	29.55	28.5	28.7	6517100	-2.05
KDH	2026-01-22	28.7	30.3	28.1	29.3	10156500	2.09
KDH	2026-01-23	29.55	29.55	28.7	28.9	2905900	-1.37
KDH	2026-01-26	29.1	29.1	27.5	27.65	5461600	-4.33
KDH	2026-01-27	27.7	27.8	26.55	26.7	7971400	-3.44
KDH	2026-01-28	26.9	27.4	25.9	26.9	8258500	0.75
KDH	2026-01-29	27	27.35	26.5	26.9	3483500	0
KDH	2026-01-30	27.1	28	26.85	27.5	11299600	2.23
\.


--
-- Data for Name: kos; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.kos (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
KOS	2025-12-26	38.8	38.8	38.25	38.4	397600	0
KOS	2025-12-29	38.8	38.8	38.4	38.5	401800	0.26
KOS	2025-12-30	38.7	38.7	38.45	38.55	356300	0.13
KOS	2025-12-31	38.9	38.9	38.55	38.55	294100	0
KOS	2026-01-05	38.55	38.7	38.55	38.6	387900	0.13
KOS	2026-01-06	38.6	38.7	38.6	38.65	423100	0.13
KOS	2026-01-07	38.65	38.75	38.65	38.65	390400	0
KOS	2026-01-08	38.65	39	38.65	39	388600	0.91
KOS	2026-01-09	39.1	39.25	39	39.05	386300	0.13
KOS	2026-01-12	39.4	39.5	39.05	39.1	390000	0.13
KOS	2026-01-13	39.1	39.15	38.8	38.8	438200	-0.77
KOS	2026-01-14	38.8	38.9	38.8	38.85	375400	0.13
KOS	2026-01-15	38.85	39.15	38.85	38.9	396900	0.13
KOS	2026-01-16	38.9	39.2	38.9	38.95	369300	0.13
KOS	2026-01-19	38.95	39	38.95	38.95	344800	0
KOS	2026-01-20	38.95	39.3	38.95	39	430100	0.13
KOS	2026-01-21	39.05	39.15	39	39	390800	0
KOS	2026-01-22	39.1	39.15	39	39.05	396900	0.13
KOS	2026-01-23	39.1	39.15	39.05	39.05	391000	0
KOS	2026-01-26	39.2	39.2	39.05	39.05	393600	0
KOS	2026-01-27	39.5	39.5	39.05	39.1	402900	0.13
KOS	2026-01-28	39.1	39.2	39.1	39.15	392000	0.13
KOS	2026-01-29	39.6	39.6	39.05	39.15	565800	0
KOS	2026-01-30	39.2	39.2	39	39.15	361400	0
\.


--
-- Data for Name: lpb; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.lpb (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
LPB	2025-12-26	42.1	42.5	41.5	41.5	2093000	0
LPB	2025-12-29	41.5	42.15	41.45	41.8	1171100	0.72
LPB	2025-12-30	41.95	41.95	41.6	41.9	817300	0.24
LPB	2025-12-31	42.05	42.05	41.45	41.8	2465100	-0.24
LPB	2026-01-05	42	42.3	39.9	40.5	4157300	-3.11
LPB	2026-01-06	40.7	41.85	40.55	41.7	4415000	2.96
LPB	2026-01-07	41.9	42.5	41.65	42.1	1271400	0.96
LPB	2026-01-08	42.2	42.3	41.4	42	1923900	-0.24
LPB	2026-01-09	42.1	42.5	40.95	41.2	2099000	-1.9
LPB	2026-01-12	41.4	42.45	41.3	42.4	2299400	2.91
LPB	2026-01-13	42.8	42.8	41.3	41.9	2377800	-1.18
LPB	2026-01-14	42.1	42.2	41.25	41.7	1518900	-0.48
LPB	2026-01-15	41.85	42.1	41	41.5	1812900	-0.48
LPB	2026-01-16	41.6	42.15	41.2	41.75	1930100	0.6
LPB	2026-01-19	41.9	42.1	41.35	41.55	863600	-0.48
LPB	2026-01-20	41.7	43.25	41.35	42.35	2707000	1.93
LPB	2026-01-21	42.5	45	42.5	42.95	3086500	1.42
LPB	2026-01-22	42.95	44	42.9	43.3	2487900	0.81
LPB	2026-01-23	43.9	43.9	41.9	42.4	1914200	-2.08
LPB	2026-01-26	42.6	43.3	41.5	41.9	1421000	-1.18
LPB	2026-01-27	42	42.55	41.9	42	1433100	0.24
LPB	2026-01-28	42.2	42.5	41.75	42	992500	0
LPB	2026-01-29	42.15	42.45	41.5	41.7	801400	-0.71
LPB	2026-01-30	41.9	42.6	41.5	41.65	1478700	-0.12
\.


--
-- Data for Name: mbb; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.mbb (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
MBB	2025-12-26	24.7	25	24.2	24.85	24369500	0
MBB	2025-12-29	24.85	25	24.7	24.85	11448200	0
MBB	2025-12-30	24.9	25.2	24.85	25	12056900	0.6
MBB	2025-12-31	25	25.3	24.95	25.3	23824800	1.2
MBB	2026-01-05	25.35	25.75	24	25.35	36858500	0.2
MBB	2026-01-06	25.45	26.75	25	26.65	72144100	5.13
MBB	2026-01-07	26.75	26.85	26.45	26.7	33897900	0.19
MBB	2026-01-08	26.75	27.15	26.5	26.6	47534900	-0.37
MBB	2026-01-09	26.75	27.7	26.7	27.3	66065300	2.63
MBB	2026-01-12	27.55	28.2	27.55	28.2	56968300	3.3
MBB	2026-01-13	28.25	28.3	27.45	27.55	45769000	-2.3
MBB	2026-01-14	27.6	27.95	26.85	27.15	45126300	-1.45
MBB	2026-01-15	27.05	27.55	26.8	27.25	34596800	0.37
MBB	2026-01-16	27.4	27.7	27.05	27.05	20359500	-0.73
MBB	2026-01-19	27.2	27.6	27.15	27.5	27218600	1.66
MBB	2026-01-20	27.75	28	27.1	27.4	28109000	-0.36
MBB	2026-01-21	27.3	27.5	26.9	27	27066700	-1.46
MBB	2026-01-22	27.15	27.45	26.9	26.95	17506700	-0.19
MBB	2026-01-23	26.95	27.1	26.65	26.95	23036800	0
MBB	2026-01-26	26.75	26.85	26	26.1	31708000	-3.15
MBB	2026-01-27	26.15	26.6	26.1	26.5	19201000	1.53
MBB	2026-01-28	26.55	26.75	26.2	26.55	17954900	0.19
MBB	2026-01-29	26.9	26.95	26.3	26.7	19308300	0.56
MBB	2026-01-30	26.75	27.2	26.6	27.2	28064600	1.87
\.


--
-- Data for Name: msb; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.msb (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
MSB	2025-12-26	12.4	12.4	12.15	12.3	8374700	0
MSB	2025-12-29	12.3	12.4	12.25	12.3	1778400	0
MSB	2025-12-30	12.3	12.5	12.3	12.5	5698600	1.63
MSB	2025-12-31	12.55	12.55	12.25	12.4	4949500	-0.8
MSB	2026-01-05	12.4	12.45	12.25	12.3	5454400	-0.81
MSB	2026-01-06	12.3	12.45	12.2	12.4	4765200	0.81
MSB	2026-01-07	12.4	12.7	12.4	12.6	8025300	1.61
MSB	2026-01-08	12.6	12.9	12.5	12.65	14636400	0.4
MSB	2026-01-09	12.7	12.85	12.5	12.5	7926600	-1.19
MSB	2026-01-12	12.55	12.95	12.5	12.9	13536800	3.2
MSB	2026-01-13	13	13	12.8	12.85	12761800	-0.39
MSB	2026-01-14	12.85	12.9	12.55	12.6	11806700	-1.95
MSB	2026-01-15	12.65	12.7	12.5	12.6	8609300	0
MSB	2026-01-16	12.6	12.75	12.55	12.55	6054000	-0.4
MSB	2026-01-19	12.6	12.65	12.5	12.65	5227500	0.8
MSB	2026-01-20	12.7	12.8	12.55	12.55	5285700	-0.79
MSB	2026-01-21	12.45	12.6	12.4	12.5	5947700	-0.4
MSB	2026-01-22	12.5	12.6	12.5	12.6	5185900	0.8
MSB	2026-01-23	12.6	12.65	12.45	12.45	3759800	-1.19
MSB	2026-01-26	12.45	12.5	12.2	12.25	8587800	-1.61
MSB	2026-01-27	12.3	12.3	12.15	12.2	4256500	-0.41
MSB	2026-01-28	12.25	12.3	12.1	12.15	4659500	-0.41
MSB	2026-01-29	12.15	12.25	12.1	12.15	2956200	0
MSB	2026-01-30	12.2	12.4	12.2	12.4	5283700	2.06
\.


--
-- Data for Name: msn; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.msn (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
MSN	2025-12-26	76	76	73.1	75.3	6065600	0
MSN	2025-12-29	75.3	75.5	74.6	75.5	2513400	0.27
MSN	2025-12-30	75.7	76.9	75.5	76.9	4153700	1.85
MSN	2025-12-31	76.7	77.2	76.1	77	4728800	0.13
MSN	2026-01-05	76.9	77	75	76.8	4140400	-0.26
MSN	2026-01-06	76.7	77.2	75.9	77	3765300	0.26
MSN	2026-01-07	77.1	78.4	76.7	78.4	8379800	1.82
MSN	2026-01-08	79	79	77.7	78.3	4525700	-0.13
MSN	2026-01-09	78.5	78.6	76.5	76.5	5786600	-2.3
MSN	2026-01-12	76.6	78.8	75.9	78.6	8295600	2.75
MSN	2026-01-13	79.5	82.2	79.3	79.3	10133600	0.89
MSN	2026-01-14	79.1	82	78.6	80.6	13858700	1.64
MSN	2026-01-15	80.5	82.5	79.8	81	9103300	0.5
MSN	2026-01-16	81.3	85.5	79.9	81.4	15818700	0.49
MSN	2026-01-19	82.1	82.3	80.2	80.2	6226700	-1.47
MSN	2026-01-20	80.2	81.4	79.1	80	11205300	-0.25
MSN	2026-01-21	79	79.9	78.5	79.9	6385500	-0.12
MSN	2026-01-22	79.6	80.6	79.1	79.9	5049500	0
MSN	2026-01-23	80	80.2	79	79	4123600	-1.13
MSN	2026-01-26	79.1	79.5	76	77.3	5784500	-2.15
MSN	2026-01-27	77.4	77.5	76.7	76.7	3805800	-0.78
MSN	2026-01-28	77.2	79.5	77.1	79.5	8819900	3.65
MSN	2026-01-29	80.3	85	80.2	84.1	24722400	5.79
MSN	2026-01-30	84.6	85	83.3	84	10777800	-0.12
\.


--
-- Data for Name: mwg; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.mwg (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
MWG	2025-12-26	84.2	87	82.6	87	9310200	0
MWG	2025-12-29	86.2	87.7	86.1	87.1	6228800	0.11
MWG	2025-12-30	87	88.5	86.6	88.5	7753100	1.61
MWG	2025-12-31	88	88.4	87.6	88.4	4804600	-0.11
MWG	2026-01-05	87.9	88.2	85.8	87.4	5947300	-1.13
MWG	2026-01-06	86.8	89.3	86.8	88.8	6390700	1.6
MWG	2026-01-07	89	89.9	88.3	89.8	4998000	1.13
MWG	2026-01-08	89.8	89.8	86.2	87.5	10543900	-2.56
MWG	2026-01-09	86.4	87.5	85.6	86	6472200	-1.71
MWG	2026-01-12	86	87.5	85	87.5	8471500	1.74
MWG	2026-01-13	87.5	88.7	86.2	87.1	6657100	-0.46
MWG	2026-01-14	87.1	87.3	85.1	86	6951400	-1.26
MWG	2026-01-15	85	86.9	84	84	7636600	-2.33
MWG	2026-01-16	83.8	87.4	83.8	87	7671600	3.57
MWG	2026-01-19	87.4	87.4	85.5	86.7	3055300	-0.34
MWG	2026-01-20	86.7	88.3	86	86	6841300	-0.81
MWG	2026-01-21	85.9	85.9	84.4	85.4	5383600	-0.7
MWG	2026-01-22	85	86.6	85	86.5	5450600	1.29
MWG	2026-01-23	86.6	87.7	85.8	85.8	6722800	-0.81
MWG	2026-01-26	85.8	85.9	83.9	84.1	6030600	-1.98
MWG	2026-01-27	84.2	85.8	84	85	5418600	1.07
MWG	2026-01-28	84.9	85.8	84.3	85.5	6085200	0.59
MWG	2026-01-29	85.7	89.5	85.3	89.5	15892000	4.68
MWG	2026-01-30	90	93.4	89.8	92.9	20845900	3.8
\.


--
-- Data for Name: nab; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.nab (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
NAB	2025-12-26	14.4	14.4	14.15	14.2	1226800	0
NAB	2025-12-29	14.2	14.35	14.15	14.25	1248000	0.35
NAB	2025-12-30	14.25	14.3	14.15	14.3	1118200	0.35
NAB	2025-12-31	14.2	14.3	14.2	14.3	1244200	0
NAB	2026-01-05	14.3	14.35	14.15	14.2	1326900	-0.7
NAB	2026-01-06	14.2	14.3	14.15	14.2	1252000	0
NAB	2026-01-07	14.2	14.5	14.2	14.4	1112500	1.41
NAB	2026-01-08	14.4	14.8	14.4	14.6	1943400	1.39
NAB	2026-01-09	14.8	14.8	14.6	14.65	1573300	0.34
NAB	2026-01-12	14.65	15.1	14.6	15.1	2636000	3.07
NAB	2026-01-13	15.2	15.3	14.9	14.9	1811000	-1.32
NAB	2026-01-14	15.15	15.25	14.85	15	2132500	0.67
NAB	2026-01-15	15	15	14.8	14.85	1462000	-1
NAB	2026-01-16	14.8	15	14.8	14.85	1288900	0
NAB	2026-01-19	14.85	14.9	14.75	14.75	1246300	-0.67
NAB	2026-01-20	14.7	14.85	14.7	14.7	1257500	-0.34
NAB	2026-01-21	14.65	14.75	14.5	14.65	1313800	-0.34
NAB	2026-01-22	14.65	14.85	14.6	14.7	1196400	0.34
NAB	2026-01-23	14.7	14.7	14.55	14.55	1133600	-1.02
NAB	2026-01-26	14.55	14.65	14.35	14.35	1423200	-1.37
NAB	2026-01-27	14.35	14.45	14.25	14.35	1142400	0
NAB	2026-01-28	14.35	14.4	14.2	14.25	1504000	-0.7
NAB	2026-01-29	14.2	14.25	14.1	14.1	1432800	-1.05
NAB	2026-01-30	14.15	14.25	14.1	14.15	1230700	0.35
\.


--
-- Data for Name: nkg; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.nkg (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
NKG	2025-12-26	15.3	15.65	15	15.3	6410800	0
NKG	2025-12-29	15.4	15.4	15.15	15.2	1935600	-0.65
NKG	2025-12-30	15.1	15.3	15	15	2445700	-1.32
NKG	2025-12-31	15	15.1	14.85	14.85	2687800	-1
NKG	2026-01-05	14.85	15	14.4	14.55	3653300	-2.02
NKG	2026-01-06	14.55	14.8	14.25	14.45	3606200	-0.69
NKG	2026-01-07	14.5	15.1	14.5	15	3346500	3.81
NKG	2026-01-08	15.05	15.2	14.8	14.95	4511000	-0.33
NKG	2026-01-09	15	15.05	14.65	14.8	4497600	-1
NKG	2026-01-12	14.7	15.65	14.65	15.55	10265700	5.07
NKG	2026-01-13	15.8	15.8	15.35	15.5	5132100	-0.32
NKG	2026-01-14	15.55	15.95	15.35	15.6	7443300	0.65
NKG	2026-01-15	15.6	16.3	15.55	16.15	14197000	3.53
NKG	2026-01-16	16.3	16.5	15.9	16.15	7265600	0
NKG	2026-01-19	16.2	16.3	15.75	15.8	6203700	-2.17
NKG	2026-01-20	15.9	15.9	15.45	15.5	6466000	-1.9
NKG	2026-01-21	15.4	15.6	15.1	15.2	5627600	-1.94
NKG	2026-01-22	15.2	15.8	15.2	15.45	3553800	1.64
NKG	2026-01-23	15.55	15.75	15.25	15.35	2597800	-0.65
NKG	2026-01-26	15.4	15.45	14.65	14.85	5144000	-3.26
NKG	2026-01-27	14.9	14.95	14.7	14.85	2841200	0
NKG	2026-01-28	14.85	15.2	14.85	15.15	3150600	2.02
NKG	2026-01-29	15.2	15.6	15.2	15.3	4224800	0.99
NKG	2026-01-30	15.45	15.7	15.2	15.2	3550700	-0.65
\.


--
-- Data for Name: nlg; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.nlg (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
NLG	2025-12-26	31.05	31.2	29.75	30.1	3345900	0
NLG	2025-12-29	30.15	31.2	30.15	30.85	1736000	2.49
NLG	2025-12-30	30.95	31.45	30.55	30.6	1157900	-0.81
NLG	2025-12-31	30.9	30.9	30.45	30.45	1098900	-0.49
NLG	2026-01-05	30.7	31.15	30.25	31.05	2143900	1.97
NLG	2026-01-06	30.85	31.05	30.25	30.5	1492200	-1.77
NLG	2026-01-07	30.7	30.8	30.35	30.65	2017100	0.49
NLG	2026-01-08	30.7	31.4	30.55	31	4108700	1.14
NLG	2026-01-09	31.2	31.2	29.15	29.15	5587600	-5.97
NLG	2026-01-12	29.2	30.5	28.7	30	4658700	2.92
NLG	2026-01-13	31.2	31.2	30.5	30.95	3061400	3.17
NLG	2026-01-14	30.9	31.7	30.45	31	6048100	0.16
NLG	2026-01-15	31	31.8	30.6	31.2	4104200	0.65
NLG	2026-01-16	31.2	31.7	30.65	30.65	2874900	-1.76
NLG	2026-01-19	30.65	30.75	30.1	30.1	2939000	-1.79
NLG	2026-01-20	30.15	30.7	30	30.6	2366600	1.66
NLG	2026-01-21	30.7	31.05	29.6	29.8	3485800	-2.61
NLG	2026-01-22	30	31.5	29.25	30.8	5521400	3.36
NLG	2026-01-23	30.8	30.95	29.7	29.7	2793800	-3.57
NLG	2026-01-26	29.7	30	29.15	29.35	2229300	-1.18
NLG	2026-01-27	29.25	29.4	28	28.05	4560200	-4.43
NLG	2026-01-28	28	29.15	27.6	29	3461400	3.39
NLG	2026-01-29	29	29.35	28.25	28.45	2161600	-1.9
NLG	2026-01-30	28.9	29.45	28.65	29.45	3032700	3.51
\.


--
-- Data for Name: nt2; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.nt2 (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
NT2	2025-12-26	24	24.15	23.85	24.15	785900	0
NT2	2025-12-29	24.3	25	24.2	24.45	1307600	1.24
NT2	2025-12-30	24.45	24.45	24.1	24.4	494700	-0.2
NT2	2025-12-31	24.5	24.55	24.3	24.35	970700	-0.2
NT2	2026-01-05	24.35	24.9	24.15	24.35	1381900	0
NT2	2026-01-06	24.35	25	24.35	24.8	1267000	1.85
NT2	2026-01-07	24.95	25.2	24.6	25	1374100	0.81
NT2	2026-01-08	25.3	25.95	24.7	24.95	3909500	-0.2
NT2	2026-01-09	24.9	25.15	24.05	24.4	1791200	-2.2
NT2	2026-01-12	24.5	24.7	24.35	24.4	2130400	0
NT2	2026-01-13	24.45	26.05	24.45	25.9	3814700	6.15
NT2	2026-01-14	25.9	25.9	25.35	25.6	1862100	-1.16
NT2	2026-01-15	25.7	26.5	25.7	25.95	2817800	1.37
NT2	2026-01-16	26.55	26.9	25.8	25.95	2956400	0
NT2	2026-01-19	25.95	26	25.35	25.8	1404300	-0.58
NT2	2026-01-20	26.2	26.65	25.65	25.9	2515500	0.39
NT2	2026-01-21	25.9	26.5	25.6	26.1	2328100	0.77
NT2	2026-01-22	26.4	26.95	25.95	26.35	1479400	0.96
NT2	2026-01-23	26.45	26.45	25.95	26.05	1224800	-1.14
NT2	2026-01-26	25.95	26.4	25.7	26	1924200	-0.19
NT2	2026-01-27	25.9	26.8	25.9	26.65	1926600	2.5
NT2	2026-01-28	26.8	26.8	26	26.35	1103100	-1.13
NT2	2026-01-29	26.45	26.5	25.95	26.25	902600	-0.38
NT2	2026-01-30	26.4	26.7	26.15	26.3	950100	0.19
\.


--
-- Data for Name: nvl; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.nvl (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
NVL	2025-12-26	13.4	13.5	12.65	13.25	10173900	0
NVL	2025-12-29	13.55	14.15	13.55	14.15	15253500	6.79
NVL	2025-12-30	14.55	14.65	13.9	13.9	8445000	-1.77
NVL	2025-12-31	13.95	14.05	13.35	13.35	7442800	-3.96
NVL	2026-01-05	13.6	13.75	13.2	13.35	8515100	0
NVL	2026-01-06	13.5	13.7	13.3	13.35	8728100	0
NVL	2026-01-07	13.35	13.45	13.05	13.2	10517800	-1.12
NVL	2026-01-08	13.3	13.5	13.1	13.3	8042700	0.76
NVL	2026-01-09	13.25	13.3	12.5	12.55	22081700	-5.64
NVL	2026-01-12	12.6	13.05	12.2	12.8	11835400	1.99
NVL	2026-01-13	13.1	13.1	12.75	12.75	8220000	-0.39
NVL	2026-01-14	12.75	13.2	12.7	12.9	9642300	1.18
NVL	2026-01-15	12.95	13.1	12.75	12.8	5566600	-0.78
NVL	2026-01-16	12.95	13.1	12.8	13.05	8177300	1.95
NVL	2026-01-19	13.05	13.1	12.8	12.95	4809500	-0.77
NVL	2026-01-20	12.8	12.95	12.75	12.75	6554800	-1.54
NVL	2026-01-21	12.6	12.9	12.5	12.6	5163900	-1.18
NVL	2026-01-22	12.75	13.35	12.3	12.8	14204500	1.59
NVL	2026-01-23	12.95	13	12.5	12.5	7998800	-2.34
NVL	2026-01-26	12.5	12.65	11.85	11.85	12176700	-5.2
NVL	2026-01-27	12	12.15	11.4	11.4	11701500	-3.8
NVL	2026-01-28	11.65	12	11.4	11.9	6951800	4.39
NVL	2026-01-29	12	12.15	11.9	12.15	3854600	2.1
NVL	2026-01-30	12.3	13	12.15	13	13138500	7
\.


--
-- Data for Name: ocb; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ocb (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
OCB	2025-12-26	12.05	12.1	11.9	12.05	2009200	0
OCB	2025-12-29	12.15	12.15	11.95	11.95	980100	-0.83
OCB	2025-12-30	11.95	12.05	11.95	12	1136300	0.42
OCB	2025-12-31	12	12.05	11.9	11.95	979800	-0.42
OCB	2026-01-05	12	12	11.7	11.75	2524800	-1.67
OCB	2026-01-06	11.8	11.9	11.65	11.85	1477900	0.85
OCB	2026-01-07	11.95	12.05	11.9	12	1314700	1.27
OCB	2026-01-08	12	12.15	11.8	11.95	6548000	-0.42
OCB	2026-01-09	11.95	12.05	11.8	11.8	2872100	-1.26
OCB	2026-01-12	12.05	12.25	11.85	12.2	6495200	3.39
OCB	2026-01-13	12.3	12.35	12.1	12.25	3882400	0.41
OCB	2026-01-14	12.3	12.3	12	12.1	6127700	-1.22
OCB	2026-01-15	12.05	12.2	12	12.15	2669700	0.41
OCB	2026-01-16	12.2	12.2	12.05	12.1	1512600	-0.41
OCB	2026-01-19	12.1	12.15	12	12.05	1346800	-0.41
OCB	2026-01-20	12.05	12.1	11.95	12	2054700	-0.41
OCB	2026-01-21	12	12	11.85	11.95	1968100	-0.42
OCB	2026-01-22	12	12	11.9	11.95	1239600	0
OCB	2026-01-23	11.9	12	11.9	11.9	1410900	-0.42
OCB	2026-01-26	11.95	11.95	11.65	11.75	3159100	-1.26
OCB	2026-01-27	11.75	11.8	11.65	11.75	1538900	0
OCB	2026-01-28	11.75	11.8	11.65	11.7	1300900	-0.43
OCB	2026-01-29	11.7	11.8	11.7	11.7	1306600	0
OCB	2026-01-30	11.8	11.85	11.7	11.8	1483400	0.85
\.


--
-- Data for Name: pan; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.pan (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
PAN	2025-12-26	28	28.1	27.55	27.8	850800	0
PAN	2025-12-29	27.75	27.95	27.3	27.5	521000	-1.08
PAN	2025-12-30	27.75	27.8	27.3	27.6	355500	0.36
PAN	2025-12-31	27.75	27.75	27.3	27.3	228600	-1.09
PAN	2026-01-05	27.3	27.5	26.6	26.9	617000	-1.47
PAN	2026-01-06	26.9	26.9	26.05	26.7	921500	-0.74
PAN	2026-01-07	26.75	27.35	26.65	27.15	677200	1.69
PAN	2026-01-08	27.15	27.5	26.75	26.85	1358200	-1.1
PAN	2026-01-09	26.65	27.15	26.65	26.7	1512100	-0.56
PAN	2026-01-12	26.9	27.5	26.9	27.5	1243800	3
PAN	2026-01-13	27.75	28.05	27.5	27.6	1362500	0.36
PAN	2026-01-14	27.8	28.15	27.6	27.65	1064600	0.18
PAN	2026-01-15	28.05	28.6	27.75	28.2	1261400	1.99
PAN	2026-01-16	28.6	28.7	28	28.2	1769700	0
PAN	2026-01-19	28.2	28.5	28	28.1	994800	-0.35
PAN	2026-01-20	28.1	29.5	28.05	29.2	1766600	3.91
PAN	2026-01-21	29.05	29.2	28.45	28.9	826000	-1.03
PAN	2026-01-22	28.9	29.3	28.65	28.9	472600	0
PAN	2026-01-23	28.9	29.25	28.6	29	822700	0.35
PAN	2026-01-26	28.9	29.4	28.6	29.1	1436500	0.34
PAN	2026-01-27	29.4	30.2	29.3	29.85	2491800	2.58
PAN	2026-01-28	30	30	29.25	29.45	607000	-1.34
PAN	2026-01-29	29.45	29.7	29.05	29.7	1253500	0.85
PAN	2026-01-30	29.7	30.1	29.45	29.8	776900	0.34
\.


--
-- Data for Name: pc1; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.pc1 (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
PC1	2025-12-26	21.95	22.25	21.3	21.75	1947300	0
PC1	2025-12-29	22.05	23.25	22.05	22.6	5089400	3.91
PC1	2025-12-30	22.55	22.95	22.5	22.7	2374200	0.44
PC1	2025-12-31	22.8	23.2	22.55	22.55	2002600	-0.66
PC1	2026-01-05	22.65	23.4	22.4	22.9	4079900	1.55
PC1	2026-01-06	23.15	24.5	23.05	24.5	15776300	6.99
PC1	2026-01-07	25.15	25.25	24.6	25.05	5766500	2.24
PC1	2026-01-08	25.1	25.1	24.3	24.45	6744500	-2.4
PC1	2026-01-09	24.5	24.7	23.6	23.75	6590000	-2.86
PC1	2026-01-12	23.75	24.35	23.75	24.1	5425500	1.47
PC1	2026-01-13	24.3	24.85	23.8	24.15	6392300	0.21
PC1	2026-01-14	24.2	24.9	24	24.15	10686300	0
PC1	2026-01-15	24.2	25.6	24.15	25.6	12206100	6
PC1	2026-01-16	25.8	25.85	24.6	24.6	5604800	-3.91
PC1	2026-01-19	24.6	24.9	24.05	24.1	4858300	-2.03
PC1	2026-01-20	24.3	24.65	23.65	23.8	5843300	-1.24
PC1	2026-01-21	23.7	23.8	23	23.25	6595100	-2.31
PC1	2026-01-22	23.6	24.2	23.3	24.1	5547800	3.66
PC1	2026-01-23	24.1	24.1	23.35	23.35	3824800	-3.11
PC1	2026-01-26	23.25	23.65	22.45	22.6	8343900	-3.21
PC1	2026-01-27	22.65	23.1	22.4	22.85	3440800	1.11
PC1	2026-01-28	23.2	23.3	22.6	22.9	4913000	0.22
PC1	2026-01-29	22.95	24.15	22.95	24	6508000	4.8
PC1	2026-01-30	24.6	24.95	24.2	24.2	8072600	0.83
\.


--
-- Data for Name: pdr; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.pdr (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
PDR	2025-12-26	19.2	19.5	18.55	19.35	12059900	0
PDR	2025-12-29	19.5	19.75	19.3	19.6	9521200	1.29
PDR	2025-12-30	19.6	19.6	19.15	19.4	7580000	-1.02
PDR	2025-12-31	19.4	19.4	18.8	18.8	11488500	-3.09
PDR	2026-01-05	18.85	19.25	18.5	18.7	8993700	-0.53
PDR	2026-01-06	18.7	18.9	18.1	18.65	10508400	-0.27
PDR	2026-01-07	18.7	18.9	18.3	18.8	8778200	0.8
PDR	2026-01-08	18.85	19.15	18.45	18.5	10292300	-1.6
PDR	2026-01-09	18.5	18.5	17.25	17.25	34769500	-6.76
PDR	2026-01-12	17.1	18.1	16.6	17.9	15821400	3.77
PDR	2026-01-13	18.2	18.25	17.85	18	12841200	0.56
PDR	2026-01-14	18	18.3	17.5	17.75	16431400	-1.39
PDR	2026-01-15	17.7	18.3	17.3	17.85	17339500	0.56
PDR	2026-01-16	17.9	18	17.35	17.5	9290100	-1.96
PDR	2026-01-19	17.55	17.8	17.4	17.8	9928500	1.71
PDR	2026-01-20	17.85	17.85	17.3	17.5	10515500	-1.69
PDR	2026-01-21	17.4	17.6	16.85	17.4	13152400	-0.57
PDR	2026-01-22	17.4	18.5	17	18.2	19858000	4.6
PDR	2026-01-23	18	18.05	17.65	17.8	7230500	-2.2
PDR	2026-01-26	17.8	17.8	16.85	17.35	11483300	-2.53
PDR	2026-01-27	17.2	17.3	16.85	17.3	7934300	-0.29
PDR	2026-01-28	17.1	17.7	16.75	17.45	9759100	0.87
PDR	2026-01-29	17.45	17.6	17.15	17.3	4145700	-0.86
PDR	2026-01-30	17.5	17.95	17.35	17.7	10651800	2.31
\.


--
-- Data for Name: phr; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.phr (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
PHR	2025-12-26	55.5	56.5	54.6	55.9	271100	0
PHR	2025-12-29	55.9	57.4	55.9	56.2	97500	0.54
PHR	2025-12-30	56.5	60.1	56.5	59	1500500	4.98
PHR	2025-12-31	59	59.4	58.3	58.5	150700	-0.85
PHR	2026-01-05	58.4	58.8	56	56.8	664000	-2.91
PHR	2026-01-06	56.8	59.9	56.8	58.3	730400	2.64
PHR	2026-01-07	59	60.7	58.5	60	1433500	2.92
PHR	2026-01-08	61	61.7	58.4	58.4	1067300	-2.67
PHR	2026-01-09	58.6	62.4	58.6	62.4	1903200	6.85
PHR	2026-01-12	63.9	65.8	62.4	62.8	1017600	0.64
PHR	2026-01-13	63.1	65	61	63.5	1302900	1.11
PHR	2026-01-14	63.6	66.5	62.3	63.5	2066700	0
PHR	2026-01-15	63.8	67.9	63.7	67.9	2265300	6.93
PHR	2026-01-16	69.5	69.5	65.6	66	917100	-2.8
PHR	2026-01-19	65.5	69.6	65.5	68.1	1205800	3.18
PHR	2026-01-20	68	68.4	66	66	1227100	-3.08
PHR	2026-01-21	65.1	65.3	61.5	62.6	1573700	-5.15
PHR	2026-01-22	63	65	62.8	64	840400	2.24
PHR	2026-01-23	64	64	62.2	62.2	793200	-2.81
PHR	2026-01-26	62.2	64	61.1	62.5	667100	0.48
PHR	2026-01-27	63	64.2	62.2	64	595800	2.4
PHR	2026-01-28	64.1	64.8	62.3	62.7	643800	-2.03
PHR	2026-01-29	63.4	63.4	61.5	62	799000	-1.12
PHR	2026-01-30	63.1	64.9	62.2	64.3	763700	3.71
\.


--
-- Data for Name: plx; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.plx (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
PLX	2025-12-26	35.05	35.55	34.7	35.55	1963300	0
PLX	2025-12-29	35.7	37	35.6	36.5	5305400	2.67
PLX	2025-12-30	36.6	36.6	35.7	35.75	1525900	-2.05
PLX	2025-12-31	35.95	36.3	35.3	35.3	1541000	-1.26
PLX	2026-01-05	35.9	37	35.35	36.3	6072100	2.83
PLX	2026-01-06	36.35	38.8	36.3	38.75	13251500	6.75
PLX	2026-01-07	39.4	41.45	38.85	41.45	15146400	6.97
PLX	2026-01-08	44	44	40	41.5	12736300	0.12
PLX	2026-01-09	41.7	44	41.5	43	8135100	3.61
PLX	2026-01-12	44	45.55	41.55	42.4	9725000	-1.4
PLX	2026-01-13	42.4	45.35	42.3	45.35	9964700	6.96
PLX	2026-01-14	47.6	48.5	47	48.5	15641100	6.95
PLX	2026-01-15	48.1	51.8	47	51.8	15407000	6.8
PLX	2026-01-16	52	55.4	50	52	14325800	0.39
PLX	2026-01-19	52.8	55.6	51.5	55.6	11820900	6.92
PLX	2026-01-20	58.4	59.4	57.5	59	11931600	6.12
PLX	2026-01-21	57.8	60.6	55.4	59.1	12501000	0.17
PLX	2026-01-22	60	61.2	56.3	56.5	9461200	-4.4
PLX	2026-01-23	56.4	57.3	53	54.6	12328300	-3.36
PLX	2026-01-26	54.3	58.4	54.2	57	14211800	4.4
PLX	2026-01-27	58	60.9	56.5	60.9	13432600	6.84
PLX	2026-01-28	63.2	64.8	57.2	58.9	15553600	-3.28
PLX	2026-01-29	58.2	58.8	56.4	57.7	10120500	-2.04
PLX	2026-01-30	58.4	59.5	57.7	58.9	6913800	2.08
\.


--
-- Data for Name: pnj; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.pnj (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
PNJ	2025-12-26	94.29	94.98	91.12	94.98	617720	0
PNJ	2025-12-29	95.18	96.07	94.29	95.08	339718	0.11
PNJ	2025-12-30	94.98	96.46	94.78	95.87	578596	0.83
PNJ	2025-12-31	94.88	97.06	94.88	96.07	461701	0.21
PNJ	2026-01-05	96.76	97.55	95.57	97.55	552025	1.54
PNJ	2026-01-06	97.06	98.05	96.46	97.75	384831	0.21
PNJ	2026-01-07	97.75	102.01	97.46	101.62	1267095	3.96
PNJ	2026-01-08	101.62	103	101.02	103	1119978	1.36
PNJ	2026-01-09	103.2	104.2	102	103	729587	0
PNJ	2026-01-12	104	104	99.2	103.2	1198139	0.19
PNJ	2026-01-13	100.5	103.5	100.4	102.9	930936	-0.29
PNJ	2026-01-14	101.1	103	100.5	101.5	566853	-1.36
PNJ	2026-01-15	100.1	102	99.5	101.5	754710	0
PNJ	2026-01-16	101.5	108.6	101.5	107.7	4718299	6.11
PNJ	2026-01-19	107.7	115	107	114.2	2887781	6.04
PNJ	2026-01-20	112.9	114	110.7	111.9	1921169	-2.01
PNJ	2026-01-21	109.7	112.8	108.5	109.9	1997263	-1.79
PNJ	2026-01-22	110.2	111.9	107.7	110.4	1620200	0.45
PNJ	2026-01-23	110	115	108.9	112.4	2202500	1.81
PNJ	2026-01-26	113.8	117.8	112.8	116.8	3634400	3.91
PNJ	2026-01-27	116.8	117.4	115.4	116	1294600	-0.68
PNJ	2026-01-28	116.9	121.1	114	118.5	2418400	2.16
PNJ	2026-01-29	119.9	126.5	119	126.5	6501500	6.75
PNJ	2026-01-30	124.8	127	121.5	127	3736300	0.4
\.


--
-- Data for Name: pow; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.pow (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
POW	2025-12-26	12.5	12.85	12.45	12.8	9882400	0
POW	2025-12-29	12.85	13.15	12.85	13	14981100	1.56
POW	2025-12-30	13.05	13.1	12.75	12.8	4526600	-1.54
POW	2025-12-31	12.8	12.9	12.7	12.7	6650400	-0.78
POW	2026-01-05	12.75	12.9	12.5	12.65	8719400	-0.39
POW	2026-01-06	12.65	13.15	12.65	12.75	16439400	0.79
POW	2026-01-07	12.85	13.6	12.85	13.6	24435000	6.67
POW	2026-01-08	14.1	14.55	13.85	14	41937800	2.94
POW	2026-01-09	14.05	14.8	14.05	14.4	24392800	2.86
POW	2026-01-12	14.55	14.75	13.8	14.15	27214700	-1.74
POW	2026-01-13	14.15	14.95	13.9	14.6	26788100	3.18
POW	2026-01-14	14.9	15.2	14.4	14.6	30743400	0
POW	2026-01-15	14.6	14.65	14.1	14.25	16912800	-2.4
POW	2026-01-16	14.45	14.85	14.2	14.25	19219400	0
POW	2026-01-19	14.35	14.5	14.15	14.25	9798300	0
POW	2026-01-20	14.45	14.8	14.25	14.3	16837400	0.35
POW	2026-01-21	14.25	14.65	14.1	14.65	17983100	2.45
POW	2026-01-22	15	15.2	14.65	14.8	24156000	1.02
POW	2026-01-23	14.8	14.8	13.9	13.9	18342700	-6.08
POW	2026-01-26	13.95	14.3	13.75	14	17346500	0.72
POW	2026-01-27	13.9	14.2	13.6	14	15959100	0
POW	2026-01-28	14.2	14.3	13.8	13.8	17763300	-1.43
POW	2026-01-29	13.85	13.95	13.35	13.4	16579300	-2.9
POW	2026-01-30	13.55	13.85	13.5	13.75	13390500	2.61
\.


--
-- Data for Name: pvd; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.pvd (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
PVD	2025-12-26	26.45	27.55	26.45	27.35	8528900	0
PVD	2025-12-29	27.75	28.7	27.45	28.5	10738900	4.2
PVD	2025-12-30	28.5	28.6	27.65	28.45	7041900	-0.18
PVD	2025-12-31	28.4	28.75	27.8	28.35	7985900	-0.35
PVD	2026-01-05	29	29.8	28.2	29.1	12825500	2.65
PVD	2026-01-06	29	30.2	28.9	29.6	9400200	1.72
PVD	2026-01-07	29.35	30.1	29.05	30	9708800	1.35
PVD	2026-01-08	30.2	30.95	29.45	30.4	17076400	1.33
PVD	2026-01-09	30.4	31.2	29.5	29.5	7788200	-2.96
PVD	2026-01-12	29.5	29.75	27.85	28.4	14566900	-3.73
PVD	2026-01-13	28.5	30.35	28.45	30.35	13447300	6.87
PVD	2026-01-14	30.9	31.5	29.45	29.7	11872800	-2.14
PVD	2026-01-15	29.45	29.85	28.6	29.05	9502400	-2.19
PVD	2026-01-16	29.5	29.6	28.9	28.9	7201500	-0.52
PVD	2026-01-19	29	29.7	28.65	29.3	5055100	1.38
PVD	2026-01-20	29.45	29.45	28.5	28.5	6388200	-2.73
PVD	2026-01-21	28.5	29.8	28.2	29.75	7161500	4.39
PVD	2026-01-22	30	30.2	28.6	29.1	6790800	-2.18
PVD	2026-01-23	29.3	29.3	27.3	27.6	8017500	-5.15
PVD	2026-01-26	28	28.5	26.95	27.4	10634600	-0.72
PVD	2026-01-27	27.8	29.3	27.4	29.3	14281000	6.93
PVD	2026-01-28	30.35	31.35	29.6	30.5	17317200	4.1
PVD	2026-01-29	31	31.05	29.55	30	6735800	-1.64
PVD	2026-01-30	30.55	31.7	30.2	31	13853100	3.33
\.


--
-- Data for Name: pvt; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.pvt (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
PVT	2025-12-26	18.35	18.7	18.2	18.5	2005600	0
PVT	2025-12-29	18.6	19.2	18.6	18.95	3911600	2.43
PVT	2025-12-30	19.1	19.1	18.65	18.8	1876300	-0.79
PVT	2025-12-31	18.85	18.85	18.4	18.4	1956100	-2.13
PVT	2026-01-05	18.6	19.1	18.4	18.8	3823400	2.17
PVT	2026-01-06	18.85	19.4	18.8	19.2	7416900	2.13
PVT	2026-01-07	19.3	20.5	19.25	20.5	12372000	6.77
PVT	2026-01-08	21	21.5	20.5	20.75	13036900	1.22
PVT	2026-01-09	20.65	21.35	20.2	20.3	6804500	-2.17
PVT	2026-01-12	20.75	20.75	19.5	19.7	8863000	-2.96
PVT	2026-01-13	19.85	21.05	19.7	21.05	6912700	6.85
PVT	2026-01-14	21.55	22.1	20.7	20.9	12244100	-0.71
PVT	2026-01-15	20.9	21.1	20.3	20.5	5422200	-1.91
PVT	2026-01-16	20.65	21.3	20.45	20.6	5430000	0.49
PVT	2026-01-19	20.9	21.05	20.4	20.6	3024600	0
PVT	2026-01-20	20.6	20.85	20.3	20.3	4598700	-1.46
PVT	2026-01-21	20.1	20.5	19.35	20.5	7474700	0.99
PVT	2026-01-22	20.85	21.15	20.5	20.85	7182700	1.71
PVT	2026-01-23	21	21	19.65	19.85	4295900	-4.8
PVT	2026-01-26	19.9	20.85	19.85	20.4	6395900	2.77
PVT	2026-01-27	20.4	21	19.9	20.55	3686100	0.74
PVT	2026-01-28	20.85	21.45	20.5	21	8278300	2.19
PVT	2026-01-29	21.1	21.1	20.5	20.5	2513200	-2.38
PVT	2026-01-30	20.8	21.4	20.7	20.75	4639700	1.22
\.


--
-- Data for Name: ree; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ree (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
REE	2025-12-26	60.42	60.62	60.03	60.33	302689	0
REE	2025-12-29	60.33	60.52	59.93	60.33	218388	0
REE	2025-12-30	60.33	60.52	60.03	60.23	232459	-0.17
REE	2025-12-31	60.33	60.62	60.03	60.03	354887	-0.33
REE	2026-01-05	60.33	61.01	59.83	59.83	745603	-0.33
REE	2026-01-06	60.03	60.92	59.83	60.23	380157	0.67
REE	2026-01-07	60.23	61.41	60.23	61.01	639578	1.3
REE	2026-01-08	61.01	61.41	60.52	60.72	616451	-0.48
REE	2026-01-09	61.01	61.51	60.03	60.03	783443	-1.14
REE	2026-01-12	60.33	60.72	60.03	60.52	967738	0.82
REE	2026-01-13	61.8	62.39	61.01	61.11	1178384	0.97
REE	2026-01-14	61.11	62.29	61.11	61.31	1109043	0.33
REE	2026-01-15	61.9	61.9	61.11	61.31	586189	0
REE	2026-01-16	61.31	64.46	61.31	62.69	1135178	2.25
REE	2026-01-19	62.98	64.06	62	62.29	755758	-0.64
REE	2026-01-20	62.39	62.79	61.7	61.8	764294	-0.79
REE	2026-01-21	61.7	61.8	61.01	61.11	611720	-1.12
REE	2026-01-22	62.1	62.88	61.7	62.2	826134	1.78
REE	2026-01-23	62.59	62.88	61.8	61.8	416118	-0.64
REE	2026-01-26	62.1	62.2	60.72	60.82	716447	-1.59
REE	2026-01-27	60.82	61.41	60.82	61.41	319589	0.97
REE	2026-01-28	61.8	61.8	60.72	60.82	535149	-0.96
REE	2026-01-29	61.11	62	60.82	61.01	706221	0.31
REE	2026-01-30	61.41	61.51	60.82	61.11	741542	0.16
\.


--
-- Data for Name: sab; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.sab (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
SAB	2025-12-26	46.13	47.19	46.13	46.76	1436057	0
SAB	2025-12-29	46.8	47.24	46.61	46.8	885071	0.09
SAB	2025-12-30	46.9	47.86	46.42	47.57	1281754	1.65
SAB	2025-12-31	47.57	47.57	47	47	581755	-1.2
SAB	2026-01-05	47.14	47.19	46.08	46.13	1627486	-1.85
SAB	2026-01-06	46.23	46.23	45.22	45.61	1946074	-1.13
SAB	2026-01-07	45.65	46.85	45.65	46.52	2003098	2
SAB	2026-01-08	46.52	47.38	46.32	46.32	1602353	-0.43
SAB	2026-01-09	46.9	47.86	46.61	46.95	2046272	1.36
SAB	2026-01-12	47.2	47.45	46.15	47	1687846	0.11
SAB	2026-01-13	47.05	50.2	47.05	50.2	3328455	6.81
SAB	2026-01-14	51.3	53.7	51.3	53.7	4561544	6.97
SAB	2026-01-15	55	57.1	52.7	52.7	6089903	-1.86
SAB	2026-01-16	52.7	53.8	51.8	52.5	2858276	-0.38
SAB	2026-01-19	52.5	52.7	50.2	51.1	2773536	-2.67
SAB	2026-01-20	52	54.3	51.5	52.3	3639462	2.35
SAB	2026-01-21	52	52	50.2	50.6	2226464	-3.25
SAB	2026-01-22	51	52	50.5	50.6	1399834	0
SAB	2026-01-23	50.7	51.3	48.9	49.3	2328200	-2.57
SAB	2026-01-26	49.3	50	48.3	48.5	1694200	-1.62
SAB	2026-01-27	48.95	49.55	48.25	49.55	1290400	2.16
SAB	2026-01-28	49.9	50.3	48.75	48.95	1151800	-1.21
SAB	2026-01-29	49.1	50.6	49	50.3	1623900	2.76
SAB	2026-01-30	51	51.4	49.9	49.9	1724800	-0.8
\.


--
-- Data for Name: sbt; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.sbt (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
SBT	2025-12-26	24.3	24.3	23.05	24.2	510600	0
SBT	2025-12-29	24.2	24.45	24.15	24.45	872400	1.03
SBT	2025-12-30	24.45	24.7	24.45	24.5	665400	0.2
SBT	2025-12-31	24.5	25.15	24.45	25.1	1172400	2.45
SBT	2026-01-05	25.1	25.1	24.7	24.9	593900	-0.8
SBT	2026-01-06	24.8	24.8	24.5	24.5	783400	-1.61
SBT	2026-01-07	24.45	24.5	24.1	24.5	673800	0
SBT	2026-01-08	24.45	24.5	24	24.5	618000	0
SBT	2026-01-09	24.45	24.5	24.35	24.4	643300	-0.41
SBT	2026-01-12	24.4	24.45	24.3	24.35	568800	-0.2
SBT	2026-01-13	24.35	24.4	24	24	675100	-1.44
SBT	2026-01-14	24	24.35	23.95	24.25	829700	1.04
SBT	2026-01-15	24.25	24.3	24.05	24.25	820400	0
SBT	2026-01-16	24.25	24.35	24.15	24.3	504200	0.21
SBT	2026-01-19	24.3	24.45	24.2	24.3	550400	0
SBT	2026-01-20	24.3	24.4	24.25	24.35	532900	0.21
SBT	2026-01-21	24.35	24.35	24	24	511900	-1.44
SBT	2026-01-22	24	24.2	24	24	599200	0
SBT	2026-01-23	24	24.2	23.95	24.05	484600	0.21
SBT	2026-01-26	24.05	24.05	23.85	24	368100	-0.21
SBT	2026-01-27	24	24.05	23.8	24.05	501300	0.21
SBT	2026-01-28	24.05	24.1	23.9	24	446700	-0.21
SBT	2026-01-29	24	24.05	23.55	24	600400	0
SBT	2026-01-30	24	24.05	23.05	23.9	721700	-0.42
\.


--
-- Data for Name: scs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.scs (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
SCS	2025-12-26	52	52	51.5	51.6	215510	0
SCS	2025-12-29	51.8	51.8	51.2	51.2	218227	-0.78
SCS	2025-12-30	51.2	52	51	51.4	184842	0.39
SCS	2025-12-31	51.9	51.9	51.3	51.4	131079	0
SCS	2026-01-05	51.7	51.8	51.4	51.4	150500	0
SCS	2026-01-06	51.6	51.6	51.2	51.3	148000	-0.19
SCS	2026-01-07	51.2	51.9	51.2	51.8	220900	0.97
SCS	2026-01-08	52.5	54.6	52.5	53.5	636100	3.28
SCS	2026-01-09	53.5	54.5	53	53.4	252300	-0.19
SCS	2026-01-12	53.8	54.3	53.3	53.8	374000	0.75
SCS	2026-01-13	54	54	52.9	53.1	316000	-1.3
SCS	2026-01-14	53	55.9	53	55.1	754300	3.77
SCS	2026-01-15	56	56.1	54.2	55.3	525000	0.36
SCS	2026-01-16	55.4	55.7	55.1	55.3	600800	0
SCS	2026-01-19	55.4	56.3	54.2	54.2	569100	-1.99
SCS	2026-01-20	54.5	57.2	54.5	56.2	717400	3.69
SCS	2026-01-21	56	56	55	55.5	346400	-1.25
SCS	2026-01-22	56	57.9	56	57.1	879200	2.88
SCS	2026-01-23	57.2	57.2	56	56	292100	-1.93
SCS	2026-01-26	56	57	54.2	54.8	415600	-2.14
SCS	2026-01-27	54.8	54.8	53.2	53.8	520400	-1.82
SCS	2026-01-28	53.8	54.7	53.8	53.9	154400	0.19
SCS	2026-01-29	53.9	55.4	53.9	54.4	163300	0.93
SCS	2026-01-30	54.8	55.3	54.5	54.5	174000	0.18
\.


--
-- Data for Name: shb; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.shb (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
SHB	2025-12-26	16.35	16.5	15.9	16.35	108127100	0
SHB	2025-12-29	16.35	16.35	16.2	16.2	44770600	-0.92
SHB	2025-12-30	16.15	16.4	16.15	16.35	65877000	0.93
SHB	2025-12-31	16.35	16.4	16.2	16.35	52638000	0
SHB	2026-01-05	16.3	16.4	15.9	16.1	64342300	-1.53
SHB	2026-01-06	16	16.2	15.8	16.2	77500200	0.62
SHB	2026-01-07	16.25	16.55	16.2	16.5	81138200	1.85
SHB	2026-01-08	16.55	17.05	16.4	16.6	135955100	0.61
SHB	2026-01-09	16.7	16.8	16.5	16.5	68480400	-0.6
SHB	2026-01-12	16.45	17.2	16.4	17	107621400	3.03
SHB	2026-01-13	17.2	17.2	16.65	16.7	88823200	-1.76
SHB	2026-01-14	16.7	16.85	16.55	16.7	81592100	0
SHB	2026-01-15	16.65	16.7	16.25	16.5	83216100	-1.2
SHB	2026-01-16	16.6	16.8	16.35	16.35	58807900	-0.91
SHB	2026-01-19	16.4	16.65	16.3	16.5	59592800	0.92
SHB	2026-01-20	16.55	16.7	16.35	16.55	54106900	0.3
SHB	2026-01-21	16.4	16.5	16.25	16.45	65842200	-0.6
SHB	2026-01-22	16.45	16.55	16.3	16.3	58903900	-0.91
SHB	2026-01-23	16.35	16.4	16.25	16.3	49017900	0
SHB	2026-01-26	16.25	16.35	15.65	15.75	75374500	-3.37
SHB	2026-01-27	15.8	16.05	15.65	16.05	51904700	1.9
SHB	2026-01-28	15.95	16.05	15.8	16	42882400	-0.31
SHB	2026-01-29	15.9	16	15.8	15.95	41935000	-0.31
SHB	2026-01-30	15.9	16.2	15.8	16	53345400	0.31
\.


--
-- Data for Name: sip; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.sip (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
SIP	2025-12-26	51	51.7	50.7	51.3	185900	0
SIP	2025-12-29	51.3	52.4	51.3	52.1	182700	1.56
SIP	2025-12-30	52.1	52.5	51.8	51.8	125700	-0.58
SIP	2025-12-31	51.9	53.8	51.8	52.6	270600	1.54
SIP	2026-01-05	53.5	53.5	51.9	52	159700	-1.14
SIP	2026-01-06	52	53.4	51.8	52.4	308500	0.77
SIP	2026-01-07	52.5	55	52.5	54.3	430500	3.63
SIP	2026-01-08	55	55.5	53.9	54.3	408900	0
SIP	2026-01-09	54.5	56.7	54.5	56	512800	3.13
SIP	2026-01-12	56.1	56.2	55.1	55.6	447900	-0.71
SIP	2026-01-13	56	56.3	54.9	55.5	487900	-0.18
SIP	2026-01-14	55.6	57	55.5	56.5	1021800	1.8
SIP	2026-01-15	56.3	57.7	56.3	57.1	560000	1.06
SIP	2026-01-16	57.1	57.5	56.8	57.2	430000	0.18
SIP	2026-01-19	57.2	60.9	57.1	59	1323500	3.15
SIP	2026-01-20	59	59.6	57.2	57.2	800200	-3.05
SIP	2026-01-21	57.2	57.5	55.6	56.1	848000	-1.92
SIP	2026-01-22	57	58.6	56.5	58.1	794400	3.57
SIP	2026-01-23	58.6	58.6	57	58	499900	-0.17
SIP	2026-01-26	58	58.2	56	56.1	462000	-3.28
SIP	2026-01-27	55.9	57.5	55.9	57.5	346900	2.5
SIP	2026-01-28	58	58.4	56.2	57	462400	-0.87
SIP	2026-01-29	58	58.7	57.1	58.6	619600	2.81
SIP	2026-01-30	59.1	61.7	59.1	61	1722900	4.1
\.


--
-- Data for Name: sjs; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.sjs (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
SJS	2025-12-26	57	58	56.8	58	6200	0
SJS	2025-12-29	58	58.4	55	58.4	25200	0.69
SJS	2025-12-30	58	61	56.6	61	56700	4.45
SJS	2025-12-31	61	61.8	59.3	61.8	126200	1.31
SJS	2026-01-05	61	61	60	60	25900	-2.91
SJS	2026-01-06	60	60.9	58.4	60.9	36300	1.5
SJS	2026-01-07	60.9	61.3	60	60.9	70700	0
SJS	2026-01-08	61	61	59.8	59.8	71200	-1.81
SJS	2026-01-09	59.8	59.8	57	57.2	32000	-4.35
SJS	2026-01-12	56.7	58.9	56.7	57.5	49700	0.52
SJS	2026-01-13	57.4	58.6	57	57.3	28900	-0.35
SJS	2026-01-14	57	59	56.9	57.2	52000	-0.17
SJS	2026-01-15	57.2	58.8	57.1	57.5	80400	0.52
SJS	2026-01-16	57.5	58.9	57.5	57.5	63300	0
SJS	2026-01-19	57.6	58.6	57.4	57.4	31300	-0.17
SJS	2026-01-20	57.4	58.5	57	57	30800	-0.7
SJS	2026-01-21	56.5	58.2	56.5	57.5	16000	0.88
SJS	2026-01-22	57.1	57.9	57	57.9	30800	0.7
SJS	2026-01-23	57.9	58	56	57.6	20900	-0.52
SJS	2026-01-26	57.9	57.9	56.5	56.9	11300	-1.22
SJS	2026-01-27	56.9	56.9	53.8	54	66300	-5.1
SJS	2026-01-28	53	53.8	50.3	53.8	65700	-0.37
SJS	2026-01-29	52.1	53.6	51.5	53.6	39500	-0.37
SJS	2026-01-30	53.2	53.2	50.5	52	81200	-2.99
\.


--
-- Data for Name: ssb; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ssb (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
SSB	2025-12-26	17.4	17.4	17.1	17.3	2094200	0
SSB	2025-12-29	17.15	17.3	17.1	17.3	1789700	0
SSB	2025-12-30	17.3	17.3	17.1	17.2	1789000	-0.58
SSB	2025-12-31	17.25	17.45	17.05	17.45	2414500	1.45
SSB	2026-01-05	17.4	17.4	17.05	17.2	2068200	-1.43
SSB	2026-01-06	17.25	17.3	17	17.3	2070200	0.58
SSB	2026-01-07	17.3	17.35	17.15	17.35	2370600	0.29
SSB	2026-01-08	17.35	17.5	17.2	17.4	2679300	0.29
SSB	2026-01-09	17.4	17.45	17.25	17.35	2096900	-0.29
SSB	2026-01-12	17.4	17.75	17.3	17.75	2809700	2.31
SSB	2026-01-13	17.8	17.8	17.5	17.7	2656400	-0.28
SSB	2026-01-14	17.7	17.95	17.55	17.9	3407100	1.13
SSB	2026-01-15	17.9	17.95	17.65	17.95	2163500	0.28
SSB	2026-01-16	17.95	18	17.8	18	1922600	0.28
SSB	2026-01-19	18	18	17.8	18	2356200	0
SSB	2026-01-20	17.95	17.95	17.65	17.7	2623600	-1.67
SSB	2026-01-21	17.65	17.65	17.25	17.4	2311100	-1.69
SSB	2026-01-22	17.3	17.7	17.3	17.7	1974100	1.72
SSB	2026-01-23	17.55	17.7	17.4	17.65	1993100	-0.28
SSB	2026-01-26	17.6	17.6	17.2	17.35	1968900	-1.7
SSB	2026-01-27	17.35	17.5	17.2	17.45	1800800	0.58
SSB	2026-01-28	17.5	17.55	17.25	17.45	1848500	0
SSB	2026-01-29	17.45	17.45	17.25	17.45	1856000	0
SSB	2026-01-30	17.4	17.5	17.15	17.15	2376700	-1.72
\.


--
-- Data for Name: ssi; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.ssi (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
SSI	2025-12-26	30.25	31.25	29.75	30.75	31205200	0
SSI	2025-12-29	30.8	30.95	30.4	30.5	11999500	-0.81
SSI	2025-12-30	30.5	30.8	30.3	30.6	11165800	0.33
SSI	2025-12-31	30.65	30.9	30.2	30.25	13612800	-1.14
SSI	2026-01-05	30.4	30.4	28.8	29.15	31181700	-3.64
SSI	2026-01-06	29.15	29.8	28.65	29.65	24039700	1.72
SSI	2026-01-07	30	30.5	29.75	30.1	24638500	1.52
SSI	2026-01-08	30.2	31.3	30.2	30.55	43204100	1.5
SSI	2026-01-09	30.95	31.05	30.3	30.35	29015400	-0.65
SSI	2026-01-12	30.45	32.45	30.45	32.45	55916600	6.92
SSI	2026-01-13	33.55	33.9	32.65	32.75	50152400	0.92
SSI	2026-01-14	32.8	33.55	32.2	33.35	58600100	1.83
SSI	2026-01-15	33.3	33.8	32.35	32.65	42148600	-2.1
SSI	2026-01-16	32.9	33.45	32	32.65	40835700	0
SSI	2026-01-19	32.6	33.2	32.45	32.65	23355000	0
SSI	2026-01-20	33	33.3	32.25	32.6	27568900	-0.15
SSI	2026-01-21	32.25	32.5	31.4	31.7	45564500	-2.76
SSI	2026-01-22	32.05	32.55	31.7	31.9	23297600	0.63
SSI	2026-01-23	32.2	32.7	31.8	31.9	26072300	0
SSI	2026-01-26	31.75	32.4	30.8	31	25986500	-2.82
SSI	2026-01-27	31	31.4	30.85	30.85	17981400	-0.48
SSI	2026-01-28	31	31.4	30.5	30.95	24978600	0.32
SSI	2026-01-29	31.05	31.5	30.95	31.15	11080700	0.65
SSI	2026-01-30	31.2	31.6	31.1	31.15	18979800	0
\.


--
-- Data for Name: stb; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.stb (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
STB	2025-12-26	55.5	57.5	53.1	57.5	17485500	0
STB	2025-12-29	58.2	60.4	58.1	59.9	22403400	4.17
STB	2025-12-30	60.3	60.8	58.5	60	19183500	0.17
STB	2025-12-31	59.6	60.2	58	58	14589500	-3.33
STB	2026-01-05	58.2	58.3	56	57.9	18337600	-0.17
STB	2026-01-06	57.5	58.7	56.7	56.7	12705700	-2.07
STB	2026-01-07	55.6	55.9	52.8	53.4	53941300	-5.82
STB	2026-01-08	53.7	55.2	53.1	53.5	16488100	0.19
STB	2026-01-09	53.6	54.3	51.5	51.8	21551500	-3.18
STB	2026-01-12	51.5	54.3	51.1	53.8	21181100	3.86
STB	2026-01-13	53.8	55.2	53.7	54.1	18230000	0.56
STB	2026-01-14	53.9	55.1	53.8	54.1	14857600	0
STB	2026-01-15	53.8	57.8	53.7	57.8	23196000	6.84
STB	2026-01-16	59.4	60.6	57.9	58.4	18237000	1.04
STB	2026-01-19	58	58.7	57.6	58.3	8938300	-0.17
STB	2026-01-20	58.2	60.9	58.1	58.1	16114900	-0.34
STB	2026-01-21	57.6	62.1	57.2	62.1	25967900	6.88
STB	2026-01-22	63	66.4	63	63.5	17206700	2.25
STB	2026-01-23	63.2	66.1	62.6	62.6	16451000	-1.42
STB	2026-01-26	63	63.8	61.6	62	14884400	-0.96
STB	2026-01-27	62	63.3	61	61.9	12496700	-0.16
STB	2026-01-28	62	65.6	61.9	63.2	12810300	2.1
STB	2026-01-29	64	64.4	62.6	62.8	7052500	-0.63
STB	2026-01-30	62.8	65	62.7	63	9372100	0.32
\.


--
-- Data for Name: szc; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.szc (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
SZC	2025-12-26	29.65	29.65	28.3	28.9	513000	0
SZC	2025-12-29	29	29	28.7	28.7	366000	-0.69
SZC	2025-12-30	28.8	28.95	28.8	28.95	286000	0.87
SZC	2025-12-31	29.2	29.45	29.05	29.15	448500	0.69
SZC	2026-01-05	29.35	29.35	29	29.1	444600	-0.17
SZC	2026-01-06	29.15	30.3	29	29.6	703800	1.72
SZC	2026-01-07	30	30.6	29.7	30.25	866900	2.2
SZC	2026-01-08	30.5	30.5	29.8	29.85	970700	-1.32
SZC	2026-01-09	30.2	30.85	29.8	30.3	967200	1.51
SZC	2026-01-12	30.3	31.05	30.1	30.85	941000	1.82
SZC	2026-01-13	30.9	31.65	30.8	31.6	1188000	2.43
SZC	2026-01-14	31.6	32.1	31.05	31.65	1465300	0.16
SZC	2026-01-15	31.65	32.2	31.45	31.45	1132300	-0.63
SZC	2026-01-16	31.55	31.9	31.05	31.2	716400	-0.79
SZC	2026-01-19	31.7	33.35	31.65	33.05	2645500	5.93
SZC	2026-01-20	33.35	33.4	32.55	32.55	976500	-1.51
SZC	2026-01-21	32.55	32.95	31.05	31.2	1595000	-4.15
SZC	2026-01-22	31.8	32.2	31.3	31.9	737300	2.24
SZC	2026-01-23	32.2	32.3	31.2	31.25	703400	-2.04
SZC	2026-01-26	31.5	31.5	30.1	30.35	869700	-2.88
SZC	2026-01-27	30	31.1	30	31	568700	2.14
SZC	2026-01-28	31.05	31.6	30.9	30.9	500800	-0.32
SZC	2026-01-29	30.9	31.4	30.6	31.3	468600	1.29
SZC	2026-01-30	31.5	32.8	31.5	32.2	1475400	2.88
\.


--
-- Data for Name: tch; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.tch (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
TCH	2025-12-26	18.35	18.75	18	18.7	5517900	0
TCH	2025-12-29	18.55	18.6	18.25	18.5	3149500	-1.07
TCH	2025-12-30	18.4	18.45	18.15	18.3	3216800	-1.08
TCH	2025-12-31	18.2	18.3	17.9	18.3	4990600	0
TCH	2026-01-05	18.15	18.2	17.5	17.9	7408600	-2.19
TCH	2026-01-06	17.6	17.85	17	17.25	8397100	-3.63
TCH	2026-01-07	17.4	17.65	17	17.6	5299900	2.03
TCH	2026-01-08	17.6	17.7	17.1	17.1	7069700	-2.84
TCH	2026-01-09	17.1	17.1	15.95	15.95	18430400	-6.73
TCH	2026-01-12	15.95	16.6	15.5	16.3	7789900	2.19
TCH	2026-01-13	16.6	16.85	16.45	16.45	6674400	0.92
TCH	2026-01-14	16.45	16.6	15.95	15.95	9496700	-3.04
TCH	2026-01-15	15.7	16.3	15.7	15.95	7917200	0
TCH	2026-01-16	16.05	16.3	15.75	15.9	4785200	-0.31
TCH	2026-01-19	15.9	16.1	15.75	16	5439200	0.63
TCH	2026-01-20	16	16.05	15.55	15.55	7560700	-2.81
TCH	2026-01-21	15.5	15.85	15	15.1	8237900	-2.89
TCH	2026-01-22	15.15	16.15	15.1	16.05	11595200	6.29
TCH	2026-01-23	16.05	16.05	15.35	15.35	5632200	-4.36
TCH	2026-01-26	15.4	15.5	14.55	14.6	7350500	-4.89
TCH	2026-01-27	14.6	14.8	14.4	14.6	4429600	0
TCH	2026-01-28	14.75	15.5	14.05	15.5	9149500	6.16
TCH	2026-01-29	15.4	15.5	14.9	15.45	5677800	-0.32
TCH	2026-01-30	15.5	15.7	15.15	15.7	7576700	1.62
\.


--
-- Data for Name: tpb; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.tpb (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
TPB	2025-12-26	16.9	17.15	16.6	16.95	7454900	0
TPB	2025-12-29	17	17.15	16.9	16.9	4987800	-0.29
TPB	2025-12-30	17	17.3	16.95	17.15	9182200	1.48
TPB	2025-12-31	17.25	17.25	17.1	17.1	4734200	-0.29
TPB	2026-01-05	17.15	17.2	16.55	16.6	9746200	-2.92
TPB	2026-01-06	16.75	17.15	16.55	17.05	9984600	2.71
TPB	2026-01-07	17.2	17.65	17.1	17.45	24298100	2.35
TPB	2026-01-08	17.5	18.15	17.3	17.65	28555100	1.15
TPB	2026-01-09	17.9	18	17.5	17.6	23633700	-0.28
TPB	2026-01-12	17.6	18.4	17.55	18.2	31613600	3.41
TPB	2026-01-13	18.3	18.5	17.85	17.9	25027900	-1.65
TPB	2026-01-14	17.95	17.95	17.5	17.5	16879600	-2.23
TPB	2026-01-15	17.5	17.65	17.3	17.4	9922700	-0.57
TPB	2026-01-16	17.65	17.7	17.35	17.4	8678500	0
TPB	2026-01-19	17.4	17.5	17.3	17.35	5048400	-0.29
TPB	2026-01-20	17.4	17.8	17.4	17.45	19291000	0.58
TPB	2026-01-21	17.3	17.45	17.1	17.2	7502100	-1.43
TPB	2026-01-22	17.2	17.6	17.2	17.4	7320300	1.16
TPB	2026-01-23	17.4	17.5	17.2	17.25	6045100	-0.86
TPB	2026-01-26	17.15	17.25	16.8	17.1	9056800	-0.87
TPB	2026-01-27	17.05	17.25	17	17.2	7435400	0.58
TPB	2026-01-28	17.25	17.25	16.85	16.9	6350500	-1.74
TPB	2026-01-29	16.9	17.1	16.7	16.7	5243000	-1.18
TPB	2026-01-30	16.8	17.2	16.65	17.2	6437600	2.99
\.


--
-- Data for Name: vcb; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vcb (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VCB	2025-12-26	57.1	57.4	56.6	57.1	3302500	0
VCB	2025-12-29	57.4	57.5	57.1	57.1	2099700	0
VCB	2025-12-30	57.3	58.1	57.2	57.3	2958000	0.35
VCB	2025-12-31	57.4	57.9	57.3	57.5	2530800	0.35
VCB	2026-01-05	57.5	57.6	57	57.1	3591800	-0.7
VCB	2026-01-06	57.1	57.5	56.6	57.3	4173200	0.35
VCB	2026-01-07	57.6	59.9	57.5	59.6	12918300	4.01
VCB	2026-01-08	59.8	63.7	59.7	63.7	27104600	6.88
VCB	2026-01-09	65.4	68	65.4	68	32481400	6.75
VCB	2026-01-12	69.9	72.7	69.6	72.7	21973100	6.91
VCB	2026-01-13	74.3	76	71.8	74	28198200	1.79
VCB	2026-01-14	73.1	78.8	73.1	76	21213600	2.7
VCB	2026-01-15	74	74.9	71.9	71.9	23227200	-5.39
VCB	2026-01-16	73	76	72	73	13426500	1.53
VCB	2026-01-19	73	73.5	71.6	72.7	8212000	-0.41
VCB	2026-01-20	74	75	72.1	73.5	18491600	1.1
VCB	2026-01-21	72.8	73.6	71.7	72.8	11179000	-0.95
VCB	2026-01-22	73.4	75.9	70.6	71	18577500	-2.47
VCB	2026-01-23	71	71	68.5	68.6	16804700	-3.38
VCB	2026-01-26	68.9	71.1	68.7	69.6	10990600	1.46
VCB	2026-01-27	69.7	71.5	68.1	70.6	9645200	1.44
VCB	2026-01-28	71.1	72.9	69.4	69.6	12283200	-1.42
VCB	2026-01-29	69.3	70.8	68.6	69.8	7855400	0.29
VCB	2026-01-30	70.6	71.9	69.6	70.5	11480700	1
\.


--
-- Data for Name: vcg; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vcg (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VCG	2025-12-26	23.1	23.45	22.6	22.9	4287700	0
VCG	2025-12-29	22.8	23.4	22.8	22.9	3461400	0
VCG	2025-12-30	22.9	23.15	22.8	22.85	3659200	-0.22
VCG	2025-12-31	22.9	22.95	22.6	22.6	5466900	-1.09
VCG	2026-01-05	22.65	22.85	21.9	22.15	4394500	-1.99
VCG	2026-01-06	22.2	22.35	21.75	22.15	4702800	0
VCG	2026-01-07	22.25	22.6	22.2	22.45	3744200	1.35
VCG	2026-01-08	22.5	22.75	22.35	22.35	5110000	-0.45
VCG	2026-01-09	22.55	22.55	21.35	21.45	9229200	-4.03
VCG	2026-01-12	21.7	22.8	21.4	22.5	6715500	4.9
VCG	2026-01-13	22.95	23.15	22.7	22.9	6721300	1.78
VCG	2026-01-14	23	23.6	22.8	23.5	8756800	2.62
VCG	2026-01-15	23.5	23.9	23.05	23.3	5148800	-0.85
VCG	2026-01-16	23.5	23.6	23.15	23.3	4587800	0
VCG	2026-01-19	23.3	24.2	23.3	23.6	7464600	1.29
VCG	2026-01-20	23.85	23.9	23.3	23.3	4175500	-1.27
VCG	2026-01-21	23.25	23.4	22.7	22.95	5783200	-1.5
VCG	2026-01-22	23	23.7	22.8	23.1	6487800	0.65
VCG	2026-01-23	23.15	23.4	22.7	22.8	4409700	-1.3
VCG	2026-01-26	22.6	22.7	21.25	21.25	24720300	-6.8
VCG	2026-01-27	19.8	20.2	19.8	19.8	38214200	-6.82
VCG	2026-01-28	19.5	20.25	18.95	19.45	16840600	-1.77
VCG	2026-01-29	19.45	19.7	18.9	19	9405400	-2.31
VCG	2026-01-30	18.95	19.45	18.85	19.15	7761900	0.79
\.


--
-- Data for Name: vci; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vci (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VCI	2025-12-26	34	35.38	33.61	35.09	14219056	0
VCI	2025-12-29	35.09	35.73	34.79	34.79	6892857	-0.85
VCI	2025-12-30	35.28	35.58	34.79	34.89	6366199	0.29
VCI	2025-12-31	34.99	35.38	34.5	34.79	8097918	-0.29
VCI	2026-01-05	34.79	34.89	33.12	33.46	10468531	-3.82
VCI	2026-01-06	33.61	34.1	33.07	33.51	7683846	0.15
VCI	2026-01-07	33.95	34.59	33.61	34.15	5886540	1.91
VCI	2026-01-08	34.4	35	34	34	11800822	-0.44
VCI	2026-01-09	34.4	34.45	33.5	33.55	7735444	-1.32
VCI	2026-01-12	33.7	35.85	33.7	35.85	21858744	6.86
VCI	2026-01-13	36.3	36.6	35.1	35.65	14466132	-0.56
VCI	2026-01-14	35.6	35.95	34.55	35.5	15049460	-0.42
VCI	2026-01-15	35.5	35.85	34.6	34.85	11414440	-1.83
VCI	2026-01-16	35.1	35.55	34	34.7	11426179	-0.43
VCI	2026-01-19	34.9	35.3	34.7	34.8	6516765	0.29
VCI	2026-01-20	35.35	36.05	34.95	35.35	12071936	1.58
VCI	2026-01-21	35.05	35.35	34.3	35	10598500	-0.99
VCI	2026-01-22	35.2	35.95	35.1	35.15	10034800	0.43
VCI	2026-01-23	35.7	37.2	35.2	35.9	23579600	2.13
VCI	2026-01-26	36.3	37.65	36	36.8	23942900	2.51
VCI	2026-01-27	36.8	37.4	36.35	37	15537000	0.54
VCI	2026-01-28	37.25	37.25	35.95	36.15	12939200	-2.3
VCI	2026-01-29	36.3	36.55	36.05	36.15	5993000	0
VCI	2026-01-30	36.15	36.65	36	36.65	9459900	1.38
\.


--
-- Data for Name: vgc; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vgc (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VGC	2025-12-26	41.15	41.2	40.15	40.85	720200	0
VGC	2025-12-29	40.95	41.3	40.7	41.3	403300	1.1
VGC	2025-12-30	41.35	41.7	40.9	41.3	555300	0
VGC	2025-12-31	41.5	42.6	41.5	41.85	991900	1.33
VGC	2026-01-05	41.9	42.1	40.6	41.5	725800	-0.84
VGC	2026-01-06	41.15	42.65	41	42	972500	1.2
VGC	2026-01-07	42	44.5	41.9	44	1609400	4.76
VGC	2026-01-08	44.4	44.6	43	43.5	1639300	-1.14
VGC	2026-01-09	43.7	46.2	43.35	44.8	2142400	2.99
VGC	2026-01-12	45.9	46.65	45.35	45.6	1631500	1.79
VGC	2026-01-13	46.05	47.8	45.1	46.5	2071600	1.97
VGC	2026-01-14	47	49.55	46.5	48.6	2880900	4.52
VGC	2026-01-15	49.3	49.8	48	48.9	1974800	0.62
VGC	2026-01-16	48.9	49.2	47.6	48	1536400	-1.84
VGC	2026-01-19	48.25	50.5	48.25	49.95	2020300	4.06
VGC	2026-01-20	50.2	50.2	48.75	48.75	1048300	-2.4
VGC	2026-01-21	48.2	48.9	47.2	47.2	1423700	-3.18
VGC	2026-01-22	47.2	48.3	47.2	47.7	1001400	1.06
VGC	2026-01-23	47.8	49.3	46.8	48.3	1090300	1.26
VGC	2026-01-26	48.5	48.9	44.95	46.1	1620700	-4.55
VGC	2026-01-27	45	45.95	44.95	45	1013200	-2.39
VGC	2026-01-28	46	46.8	45.5	46.1	605600	2.44
VGC	2026-01-29	46.1	46.9	46	46.9	724800	1.74
VGC	2026-01-30	47.05	50.1	47.05	50.1	4140800	6.82
\.


--
-- Data for Name: vhc; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vhc (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VHC	2025-12-26	53.1	54	52.9	53.7	624200	0
VHC	2025-12-29	53.7	54.4	53.7	54.4	240200	1.3
VHC	2025-12-30	55.4	56.6	55	56.6	1118400	4.04
VHC	2025-12-31	56.6	56.8	56	56.1	682200	-0.88
VHC	2026-01-05	56.3	56.3	55.1	55.5	478400	-1.07
VHC	2026-01-06	55.4	56.5	55.3	55.7	358700	0.36
VHC	2026-01-07	55.9	57.2	55.6	57	811500	2.33
VHC	2026-01-08	57.4	58.2	56.7	57.1	1043400	0.18
VHC	2026-01-09	57.6	58	56.9	57.1	609200	0
VHC	2026-01-12	57.1	57.3	56.2	56.8	764400	-0.53
VHC	2026-01-13	56.9	58.7	56.9	58.6	1850500	3.17
VHC	2026-01-14	59.2	60.8	59.1	60.2	3101200	2.73
VHC	2026-01-15	60.9	60.9	58	59.5	1033000	-1.16
VHC	2026-01-16	59.8	60.7	59.3	60.1	1246200	1.01
VHC	2026-01-19	60.5	63.2	60.5	63	2654600	4.83
VHC	2026-01-20	63.2	63.8	62.4	63	1639700	0
VHC	2026-01-21	62.4	62.8	60	62	2053800	-1.59
VHC	2026-01-22	62	63.6	60.2	60.8	1305800	-1.94
VHC	2026-01-23	61.2	62.2	59	59	1244700	-2.96
VHC	2026-01-26	58.6	60.2	58.5	60	962400	1.69
VHC	2026-01-27	59.9	60	58.5	58.5	679400	-2.5
VHC	2026-01-28	59.6	61.5	58.8	61	1287000	4.27
VHC	2026-01-29	61	65	60.8	65	4208700	6.56
VHC	2026-01-30	64.7	65.6	64.3	64.7	1548000	-0.46
\.


--
-- Data for Name: vhm; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vhm (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VHM	2025-12-26	106.3	111	106.3	110	16551200	0
VHM	2025-12-29	111.9	117.7	111	117.7	7324300	7
VHM	2025-12-30	120	120.6	113	117.5	6669000	-0.17
VHM	2025-12-31	117.5	125.7	115.8	124	6830500	5.53
VHM	2026-01-05	126.8	132.6	125.2	132.6	10136200	6.94
VHM	2026-01-06	138.5	141.8	136.5	141.8	8459900	6.94
VHM	2026-01-07	150	150.9	141	149.5	9511500	5.43
VHM	2026-01-08	150.3	150.9	139.1	139.1	15234400	-6.96
VHM	2026-01-09	138.6	142.8	132.6	140	9789300	0.65
VHM	2026-01-12	139	139	130.2	130.2	9143600	-7
VHM	2026-01-13	125.7	134.9	125.7	134	9148500	2.92
VHM	2026-01-14	134.9	134.9	124.7	126.3	11385300	-5.75
VHM	2026-01-15	124	124	117.5	120	19255700	-4.99
VHM	2026-01-16	121.1	126.8	121.1	124.1	5774400	3.42
VHM	2026-01-19	126.9	126.9	120.5	125	5284700	0.73
VHM	2026-01-20	124.1	124.5	119.7	123	7006400	-1.6
VHM	2026-01-21	121	126	120.3	122.9	9321600	-0.08
VHM	2026-01-22	120.1	122.9	118.5	120.5	7971100	-1.95
VHM	2026-01-23	122.1	127.5	121.1	122.5	7272600	1.66
VHM	2026-01-26	124	124.1	116.1	118.9	4159000	-2.94
VHM	2026-01-27	117.5	118	110.6	110.6	10162000	-6.98
VHM	2026-01-28	108.1	109	102.9	104.3	15508700	-5.7
VHM	2026-01-29	104.5	107.4	104.1	107	6368800	2.59
VHM	2026-01-30	107	107.4	104	106	4714200	-0.93
\.


--
-- Data for Name: vib; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vib (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VIB	2025-12-26	17.65	17.85	17.2	17.6	8329800	0
VIB	2025-12-29	17.6	17.75	17.55	17.6	3557400	0
VIB	2025-12-30	17.6	17.8	17.6	17.7	3401200	0.57
VIB	2025-12-31	17.8	17.95	17.7	17.75	6584200	0.28
VIB	2026-01-05	17.75	17.9	17.55	17.7	4191700	-0.28
VIB	2026-01-06	17.65	17.8	17.4	17.75	5413100	0.28
VIB	2026-01-07	17.8	18.15	17.8	18.05	8724700	1.69
VIB	2026-01-08	18.1	18.2	17.9	18	12735300	-0.28
VIB	2026-01-09	18.1	18.15	17.85	17.95	10177300	-0.28
VIB	2026-01-12	17.95	18.75	17.85	18.55	16079000	3.34
VIB	2026-01-13	18.7	18.75	18.45	18.55	9889600	0
VIB	2026-01-14	18.6	18.65	18.05	18.15	18442000	-2.16
VIB	2026-01-15	18.3	18.45	18.2	18.4	6546900	1.38
VIB	2026-01-16	18.45	18.45	18.15	18.2	8113200	-1.09
VIB	2026-01-19	18.2	18.3	18.15	18.15	5557600	-0.27
VIB	2026-01-20	18.25	18.4	18.05	18.05	6538100	-0.55
VIB	2026-01-21	18	18.05	17.7	17.8	10586400	-1.39
VIB	2026-01-22	17.8	18	17.8	17.85	4862400	0.28
VIB	2026-01-23	17.95	17.95	17.75	17.8	4259400	-0.28
VIB	2026-01-26	17.8	17.9	17.35	17.45	6874300	-1.97
VIB	2026-01-27	17.45	17.55	17.35	17.4	3499100	-0.29
VIB	2026-01-28	17.4	17.65	17.35	17.45	4773000	0.29
VIB	2026-01-29	17.45	17.6	17.4	17.5	4241000	0.29
VIB	2026-01-30	17.5	17.8	17.45	17.8	6710600	1.71
\.


--
-- Data for Name: vic; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vic (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VIC	2025-12-26	147	155.2	147	155	15336300	0
VIC	2025-12-29	155	159.9	155	159.7	6667000	3.03
VIC	2025-12-30	159.7	163	151.6	163	8091800	2.07
VIC	2025-12-31	163	172	158.8	169.6	9438700	4.05
VIC	2026-01-05	169.6	174.5	166.8	173.1	6644800	2.06
VIC	2026-01-06	173.6	176	171.4	173.1	7681000	0
VIC	2026-01-07	173.1	179.5	168.6	179	7674800	3.41
VIC	2026-01-08	183.5	190	176.6	176.6	6593700	-1.34
VIC	2026-01-09	176	180.2	172.6	176	5389600	-0.34
VIC	2026-01-12	174.6	175.4	163.7	163.7	9379500	-6.99
VIC	2026-01-13	159	173.3	158	167.9	5382600	2.57
VIC	2026-01-14	171.2	171.2	159	160.2	4416000	-4.59
VIC	2026-01-15	160.1	160.2	150.9	153	8278800	-4.49
VIC	2026-01-16	153	163	153	159.9	3824600	4.51
VIC	2026-01-19	164.9	164.9	159	162	2282900	1.31
VIC	2026-01-20	162	162	154.6	161	5147800	-0.62
VIC	2026-01-21	159	164.2	158.9	160.5	3238000	-0.31
VIC	2026-01-22	157	161.4	157	161.1	3404000	0.37
VIC	2026-01-23	163.8	167.7	162.5	165.4	3317300	2.67
VIC	2026-01-26	167	167.5	159.6	159.9	2315400	-3.33
VIC	2026-01-27	160	160	150.1	151	6581300	-5.57
VIC	2026-01-28	151	154.5	140.5	140.5	10028300	-6.95
VIC	2026-01-29	138.9	142	138.5	140.5	6992500	0
VIC	2026-01-30	140.5	141.9	135.9	140.5	6881400	0
\.


--
-- Data for Name: vix; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vix (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VIX	2025-12-26	19.42	19.68	18.54	19.25	45236209	0
VIX	2025-12-29	19.42	19.59	19.17	19.25	13708096	0
VIX	2025-12-30	19.38	19.55	19.17	19.34	24753377	0.47
VIX	2025-12-31	19.47	19.47	18.92	19	19053873	-1.76
VIX	2026-01-05	19.13	19.13	17.69	17.73	81308525	-6.68
VIX	2026-01-06	17.86	18.16	17.02	17.73	43191412	0
VIX	2026-01-07	18.03	18.75	17.99	18.45	31437135	4.06
VIX	2026-01-08	18.58	19.21	18.33	18.58	47509197	0.7
VIX	2026-01-09	18.83	18.92	17.65	18.07	58494901	-2.74
VIX	2026-01-12	18.24	19.3	18.11	19.3	61581430	6.81
VIX	2026-01-13	19.85	20.61	19.8	20.61	101502765	6.79
VIX	2026-01-14	20.86	21.2	20.01	20.86	62600098	1.21
VIX	2026-01-15	21.03	21.75	20.69	21.2	70481222	1.63
VIX	2026-01-16	21.49	21.49	20.44	20.86	50512084	-1.6
VIX	2026-01-19	20.9	22.29	20.69	21.96	68138065	5.27
VIX	2026-01-20	22.13	22.13	21.15	21.58	47674463	-1.73
VIX	2026-01-21	21.28	21.32	20.1	20.56	83051459	-4.73
VIX	2026-01-22	20.73	21.07	20.27	20.77	32723352	1.02
VIX	2026-01-23	20.86	21.24	20.44	20.52	39720549	-1.2
VIX	2026-01-26	20.52	20.52	19.09	19.21	63159561	-6.38
VIX	2026-01-27	19.38	19.51	18.87	19.09	35457706	-0.62
VIX	2026-01-28	19.25	19.25	18.37	18.62	46550883	-2.46
VIX	2026-01-29	18.75	19.17	18.58	18.58	20268471	-0.21
VIX	2026-01-30	18.71	19.17	18.28	19.09	29059127	2.74
\.


--
-- Data for Name: vjc; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vjc (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VJC	2025-12-26	208.7	212.5	205	207.9	1665600	0
VJC	2025-12-29	208.4	211.2	206.7	208.2	1680400	0.14
VJC	2025-12-30	208	208.2	199.6	205.4	1702600	-1.34
VJC	2025-12-31	204.5	209	201.7	209	1395200	1.75
VJC	2026-01-05	207	209.5	203.7	207.5	1788000	-0.72
VJC	2026-01-06	207.5	215.5	205.2	212.2	2090200	2.27
VJC	2026-01-07	212.5	212.7	207.9	209.6	1487800	-1.23
VJC	2026-01-08	210	211.2	199.5	203	2229900	-3.15
VJC	2026-01-09	201.7	202.7	190	198.7	2533600	-2.12
VJC	2026-01-12	197.8	198	188	195	1966900	-1.86
VJC	2026-01-13	194	196.5	190.2	195	1639400	0
VJC	2026-01-14	194.2	195.2	186	189.4	1902700	-2.87
VJC	2026-01-15	188.5	188.5	177.5	180.5	1674500	-4.7
VJC	2026-01-16	180.5	187	179.6	184	1380500	1.94
VJC	2026-01-19	184.5	187	182.2	185	1083300	0.54
VJC	2026-01-20	185.5	186	181.3	181.8	1416900	-1.73
VJC	2026-01-21	181.5	183.5	178.1	181.8	1463200	0
VJC	2026-01-22	181	182.1	179.1	181	1307700	-0.44
VJC	2026-01-23	181.5	193.6	179.9	193.6	1510400	6.96
VJC	2026-01-26	189.7	189.9	182	182	1220900	-5.99
VJC	2026-01-27	182.2	183.5	178.8	178.8	1224400	-1.76
VJC	2026-01-28	179	180.2	166.6	171.5	1969300	-4.08
VJC	2026-01-29	172	172.9	161	164	1782400	-4.37
VJC	2026-01-30	164.3	170.5	159.4	170.5	1696000	3.96
\.


--
-- Data for Name: vnd; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vnd (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VND	2025-12-26	19.75	20.15	19.25	20	22010800	0
VND	2025-12-29	20	20.05	19.55	19.8	8883400	-1
VND	2025-12-30	19.75	20	19.5	19.7	8305800	-0.51
VND	2025-12-31	19.7	19.8	19.35	19.45	9532900	-1.27
VND	2026-01-05	19.45	19.55	18.3	18.7	21102600	-3.86
VND	2026-01-06	18.8	19.05	18.3	18.65	12230300	-0.27
VND	2026-01-07	18.8	19.3	18.7	19.3	11587900	3.49
VND	2026-01-08	19.6	19.85	19.3	19.8	17977800	2.59
VND	2026-01-09	19.95	20	19.3	19.3	13044800	-2.53
VND	2026-01-12	19.3	20.65	19.3	20.65	30436300	6.99
VND	2026-01-13	21.15	21.45	20.65	20.8	23794900	0.73
VND	2026-01-14	20.85	21	20.05	20.6	25980200	-0.96
VND	2026-01-15	20.5	20.95	20.2	20.8	24731400	0.97
VND	2026-01-16	20.85	21	20.3	20.5	17012900	-1.44
VND	2026-01-19	20.5	20.8	20.2	20.55	13499500	0.24
VND	2026-01-20	20.75	20.8	19.6	19.6	27635600	-4.62
VND	2026-01-21	19.4	19.6	18.75	19	30839400	-3.06
VND	2026-01-22	19.3	19.6	19	19.2	11001500	1.05
VND	2026-01-23	19.3	19.7	19.1	19.4	15567300	1.04
VND	2026-01-26	19.25	19.55	18.5	18.5	19080400	-4.64
VND	2026-01-27	18.5	18.85	18.45	18.65	7018100	0.81
VND	2026-01-28	18.65	18.85	18.3	18.6	13287200	-0.27
VND	2026-01-29	18.65	18.9	18.4	18.45	5707300	-0.81
VND	2026-01-30	18.45	18.7	18.4	18.5	10447600	0.27
\.


--
-- Data for Name: vnm; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vnm (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VNM	2025-12-26	61.6	62.3	61.2	61.5	2399200	0
VNM	2025-12-29	61.5	62.2	61.4	62.1	1943000	0.98
VNM	2025-12-30	62.1	62.3	61.7	61.8	1581000	-0.48
VNM	2025-12-31	61.9	62	61.2	61.2	1879600	-0.97
VNM	2026-01-05	61.3	61.5	60	60.3	2814700	-1.47
VNM	2026-01-06	60.5	60.9	60.3	60.8	2963300	0.83
VNM	2026-01-07	60.8	61.5	60.5	60.9	3228100	0.16
VNM	2026-01-08	61.1	63.4	61.1	62.2	5636400	2.13
VNM	2026-01-09	62.4	62.5	61	61	4184100	-1.93
VNM	2026-01-12	61.1	62.7	61.1	62.7	4146300	2.79
VNM	2026-01-13	62.9	64.9	62.8	63.3	6761300	0.96
VNM	2026-01-14	63.6	67.7	63.4	67.7	24838300	6.95
VNM	2026-01-15	70	72.4	69.5	71	19260600	4.87
VNM	2026-01-16	71.1	73	69.1	69.6	13372500	-1.97
VNM	2026-01-19	69.8	71	68.3	70.6	10385300	1.44
VNM	2026-01-20	71.5	75.5	71.1	73.4	21521900	3.97
VNM	2026-01-21	73	73	70	70.3	11234100	-4.22
VNM	2026-01-22	71.1	72.8	70	70.9	8055200	0.85
VNM	2026-01-23	69.4	70.1	67.2	67.2	16050400	-5.22
VNM	2026-01-26	67.4	69.3	67.4	68.9	8864800	2.53
VNM	2026-01-27	68.4	68.6	66.1	67.7	8357800	-1.74
VNM	2026-01-28	67.7	69.5	67.2	67.7	9816800	0
VNM	2026-01-29	68.5	71.1	67.9	71.1	10036800	5.02
VNM	2026-01-30	71.5	72.5	70.5	70.6	7844800	-0.7
\.


--
-- Data for Name: vpb; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vpb (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VPB	2025-12-26	28.2	28.85	27.35	28.15	27183600	0
VPB	2025-12-29	28.15	28.3	27.8	28	20347900	-0.53
VPB	2025-12-30	27.9	29.1	27.9	28.7	23124400	2.5
VPB	2025-12-31	28.7	29	28.55	28.65	22530100	-0.17
VPB	2026-01-05	28.65	28.8	27.55	27.9	16194100	-2.62
VPB	2026-01-06	27.85	28.9	27.65	28.85	21660200	3.41
VPB	2026-01-07	29.15	29.75	29.1	29.25	34203500	1.39
VPB	2026-01-08	29.3	29.75	29	29.25	29952600	0
VPB	2026-01-09	29.25	29.65	28.2	28.25	37610200	-3.42
VPB	2026-01-12	28.5	30.2	28.15	30.2	49690500	6.9
VPB	2026-01-13	31.3	31.45	30.3	30.5	42211100	0.99
VPB	2026-01-14	30.55	30.8	29.2	29.5	44749400	-3.28
VPB	2026-01-15	29.35	29.4	28.75	29	34620800	-1.69
VPB	2026-01-16	29.3	29.45	28.85	28.9	24981000	-0.34
VPB	2026-01-19	28.95	29.8	28.9	29.65	30777100	2.6
VPB	2026-01-20	30.15	30.95	29.7	29.75	48697600	0.34
VPB	2026-01-21	29.65	29.65	29	29.05	34082700	-2.35
VPB	2026-01-22	29.1	29.4	29.05	29.2	19795300	0.52
VPB	2026-01-23	29.2	29.3	28.5	28.5	31092900	-2.4
VPB	2026-01-26	28.6	28.65	27.6	27.75	32470600	-2.63
VPB	2026-01-27	27.8	28.1	27.7	27.8	12846500	0.18
VPB	2026-01-28	27.95	28.3	27.65	27.75	14212000	-0.18
VPB	2026-01-29	28	28.15	27.65	27.65	8246800	-0.36
VPB	2026-01-30	27.75	28	27.55	28	15839700	1.27
\.


--
-- Data for Name: vpi; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vpi (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VPI	2025-12-26	61.1	61.1	59.4	59.5	2118900	0
VPI	2025-12-29	59.5	59.6	57.9	58	1854200	-2.52
VPI	2025-12-30	58	58.5	57.5	58.3	1597300	0.52
VPI	2025-12-31	58	58.1	57.6	57.9	1862500	-0.69
VPI	2026-01-05	57.9	58.6	57.4	57.5	1757600	-0.69
VPI	2026-01-06	57.7	58	55.6	56	2260400	-2.61
VPI	2026-01-07	55.9	57.3	55.6	57	2244700	1.79
VPI	2026-01-08	57.1	57.6	55.5	55.6	2044900	-2.46
VPI	2026-01-09	55.6	55.7	53.8	54.2	2189200	-2.52
VPI	2026-01-12	54.1	55.6	54	55.6	3049000	2.58
VPI	2026-01-13	55.1	55.3	54	54	2198200	-2.88
VPI	2026-01-14	54.5	54.6	52.5	53	2423800	-1.85
VPI	2026-01-15	52.9	54	52.3	53.5	2756800	0.94
VPI	2026-01-16	53.3	54.4	53.3	54.3	1968800	1.5
VPI	2026-01-19	54.3	55.1	54	54.9	2225100	1.1
VPI	2026-01-20	54.9	55.7	54.6	55.2	1893900	0.55
VPI	2026-01-21	55	55.3	54.2	54.9	2018800	-0.54
VPI	2026-01-22	54.6	56.6	54.5	55.6	1522200	1.28
VPI	2026-01-23	55.2	55.8	54.5	54.7	1268600	-1.62
VPI	2026-01-26	54.7	54.9	53.4	53.8	1128700	-1.65
VPI	2026-01-27	53.5	53.8	52.5	53.5	1404300	-0.56
VPI	2026-01-28	52.5	54	52.1	53.1	1331000	-0.75
VPI	2026-01-29	53.2	53.4	52.4	53	1659100	-0.19
VPI	2026-01-30	52.3	54.1	52.2	53.9	1869800	1.7
\.


--
-- Data for Name: vpl; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vpl (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VPL	2025-12-26	92.1	95	92.1	92.1	1119700	0
VPL	2025-12-29	92.1	97	91.6	94	616600	2.06
VPL	2025-12-30	93.1	94	89.7	91	977300	-3.19
VPL	2025-12-31	91.5	95.8	88	94.2	1038900	3.52
VPL	2026-01-05	96.3	100.7	96	100	1927300	6.16
VPL	2026-01-06	102	105.8	101.1	102.9	1478400	2.9
VPL	2026-01-07	105.9	105.9	98	100.5	1477800	-2.33
VPL	2026-01-08	102	102	94.1	94.1	1616800	-6.37
VPL	2026-01-09	92.1	97	91.9	93.1	931000	-1.06
VPL	2026-01-12	99	99	87.9	93.1	1484500	0
VPL	2026-01-13	93.1	98.5	93.1	94.9	1040900	1.93
VPL	2026-01-14	94.9	94.9	90.9	92.5	777200	-2.53
VPL	2026-01-15	90.1	92.5	89.2	92.3	761900	-0.22
VPL	2026-01-16	92.1	95.6	92.1	93	988800	0.76
VPL	2026-01-19	95.4	95.4	91.5	93.5	532100	0.54
VPL	2026-01-20	93.6	93.6	91.2	91.7	756000	-1.93
VPL	2026-01-21	91.7	93	90	91.7	1009800	0
VPL	2026-01-22	95	95	92	92	562800	0.33
VPL	2026-01-23	92.2	95.5	92.2	93.8	689900	1.96
VPL	2026-01-26	95	95	92.1	93.5	603900	-0.32
VPL	2026-01-27	93.5	93.8	91	91.9	617800	-1.71
VPL	2026-01-28	91.9	91.9	85.6	91	1341900	-0.98
VPL	2026-01-29	89.9	92	88.2	92	665100	1.1
VPL	2026-01-30	93	93.2	90.9	92.5	783500	0.54
\.


--
-- Data for Name: vre; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vre (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VRE	2025-12-26	30	32	30	32	20927400	0
VRE	2025-12-29	32	33.95	31.3	33.2	7573400	3.75
VRE	2025-12-30	33.5	33.6	32	32.8	8839700	-1.2
VRE	2025-12-31	32.5	33.8	32.2	33.65	9567900	2.59
VRE	2026-01-05	33.9	35.85	33.8	35.6	13719300	5.79
VRE	2026-01-06	36.1	38.05	35.7	38.05	21287500	6.88
VRE	2026-01-07	38.8	39.5	36.6	38.55	11861100	1.31
VRE	2026-01-08	38.4	38.5	35.9	35.9	16566300	-6.87
VRE	2026-01-09	35.6	36.05	34.5	35	15581100	-2.51
VRE	2026-01-12	34.5	35	32.55	32.6	25316700	-6.86
VRE	2026-01-13	32.35	34.45	32.35	33.55	10555700	2.91
VRE	2026-01-14	33.55	33.55	31.45	31.8	16104600	-5.22
VRE	2026-01-15	31.5	31.5	29.8	31.45	16477000	-1.1
VRE	2026-01-16	31.6	32.5	31.6	32	9250200	1.75
VRE	2026-01-19	32.55	32.6	31.15	32	8573500	0
VRE	2026-01-20	31.65	32	31	31.15	10790100	-2.66
VRE	2026-01-21	31.1	31.75	31	31.25	6966700	0.32
VRE	2026-01-22	31.2	32.15	31	31.85	7252900	1.92
VRE	2026-01-23	32.15	33	31.85	32.4	7919100	1.73
VRE	2026-01-26	32.5	32.5	30.75	31	6004100	-4.32
VRE	2026-01-27	31	31	29.2	30.9	10746100	-0.32
VRE	2026-01-28	30.45	30.8	28.75	29.1	14010100	-5.83
VRE	2026-01-29	29.85	30	29.05	30	3830500	3.09
VRE	2026-01-30	29.9	30.2	29.5	30.2	6955000	0.67
\.


--
-- Data for Name: vsc; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vsc (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VSC	2025-12-26	20.55	20.7	19.75	20.15	7047700	0
VSC	2025-12-29	20.2	20.6	20.15	20.3	3223000	0.74
VSC	2025-12-30	20.3	20.55	20.25	20.25	3171900	-0.25
VSC	2025-12-31	20.35	20.45	19.9	19.95	4873800	-1.48
VSC	2026-01-05	20.05	20.1	19	19.6	7095600	-1.75
VSC	2026-01-06	19.5	20	19.1	19.7	4849800	0.51
VSC	2026-01-07	19.95	20.55	19.85	20.55	8305800	4.31
VSC	2026-01-08	20.8	21	20.45	20.6	8007700	0.24
VSC	2026-01-09	20.65	20.7	19.65	19.75	8290000	-4.13
VSC	2026-01-12	19.7	20.7	19.2	20.55	9266800	4.05
VSC	2026-01-13	20.75	21.95	20.55	21.95	20615400	6.81
VSC	2026-01-14	22.2	22.9	21.65	22.25	14084700	1.37
VSC	2026-01-15	22.05	22.55	21.6	21.95	9932700	-1.35
VSC	2026-01-16	22.05	22.45	21.75	21.75	8876400	-0.91
VSC	2026-01-19	21.55	22.35	21.5	21.8	5828200	0.23
VSC	2026-01-20	22.05	23.3	21.95	23.3	18937700	6.88
VSC	2026-01-21	23.4	24	22.5	22.5	16271000	-3.43
VSC	2026-01-22	22.8	23.4	22.3	22.55	8567000	0.22
VSC	2026-01-23	22.7	22.85	21.5	21.85	9390200	-3.1
VSC	2026-01-26	21.85	21.85	20.35	20.35	12339600	-6.86
VSC	2026-01-27	20.4	20.65	19.95	20	5417100	-1.72
VSC	2026-01-28	20.15	20.5	19.85	20.1	5254500	0.5
VSC	2026-01-29	20.1	20.75	20.1	20.55	4542400	2.24
VSC	2026-01-30	20.7	21.05	20.55	20.6	5121600	0.24
\.


--
-- Data for Name: vtp; Type: TABLE DATA; Schema: public; Owner: postgres
--

COPY public.vtp (symbol, "time", open, high, low, close, volume, percent_change) FROM stdin;
VTP	2025-12-26	97	97.3	95.3	97	358100	0
VTP	2025-12-29	97.8	97.8	96.5	97	207700	0
VTP	2025-12-30	97.5	98.3	96.9	98.1	315400	1.13
VTP	2025-12-31	98.6	98.8	97.7	98.8	245800	0.71
VTP	2026-01-05	98.8	99.6	96.4	96.5	359600	-2.33
VTP	2026-01-06	96.8	96.8	95.3	95.6	478800	-0.93
VTP	2026-01-07	95.5	102.2	95.3	102.2	1586300	6.9
VTP	2026-01-08	105.7	105.7	99.7	100.1	945900	-2.05
VTP	2026-01-09	101	107.1	101	107.1	1384900	6.99
VTP	2026-01-12	114.5	114.5	114.5	114.5	684500	6.91
VTP	2026-01-13	122	122	110.2	115	3106000	0.44
VTP	2026-01-14	118	123	115.4	123	3882400	6.96
VTP	2026-01-15	126	131.5	123.5	128.9	1618500	4.8
VTP	2026-01-16	124.5	128.9	121.8	122	1917600	-5.35
VTP	2026-01-19	123.3	124.2	118.1	122	1588400	0
VTP	2026-01-20	123.5	129.9	122	125.5	1794400	2.87
VTP	2026-01-21	122.5	124.8	119.5	122	1228500	-2.79
VTP	2026-01-22	125	125	120.7	121	1317800	-0.82
VTP	2026-01-23	120.9	120.9	113.2	114.7	1410100	-5.21
VTP	2026-01-26	114.8	117.5	111.1	113	1131100	-1.48
VTP	2026-01-27	114.7	116.1	112	115.6	815700	2.3
VTP	2026-01-28	117.9	119.5	115	115.4	1244200	-0.17
VTP	2026-01-29	115.3	115.3	111.3	111.4	1126100	-3.47
VTP	2026-01-30	111.4	113.7	111.1	111.2	974600	-0.18
\.


--
-- Name: acb acb_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.acb
    ADD CONSTRAINT acb_pkey PRIMARY KEY ("time");


--
-- Name: anv anv_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.anv
    ADD CONSTRAINT anv_pkey PRIMARY KEY ("time");


--
-- Name: bcm bcm_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bcm
    ADD CONSTRAINT bcm_pkey PRIMARY KEY ("time");


--
-- Name: bid bid_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bid
    ADD CONSTRAINT bid_pkey PRIMARY KEY ("time");


--
-- Name: bmp bmp_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bmp
    ADD CONSTRAINT bmp_pkey PRIMARY KEY ("time");


--
-- Name: bsi bsi_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bsi
    ADD CONSTRAINT bsi_pkey PRIMARY KEY ("time");


--
-- Name: bsr bsr_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bsr
    ADD CONSTRAINT bsr_pkey PRIMARY KEY ("time");


--
-- Name: bvh bvh_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bvh
    ADD CONSTRAINT bvh_pkey PRIMARY KEY ("time");


--
-- Name: bwe bwe_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.bwe
    ADD CONSTRAINT bwe_pkey PRIMARY KEY ("time");


--
-- Name: cii cii_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cii
    ADD CONSTRAINT cii_pkey PRIMARY KEY ("time");


--
-- Name: cmg cmg_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cmg
    ADD CONSTRAINT cmg_pkey PRIMARY KEY ("time");


--
-- Name: company_info company_info_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.company_info
    ADD CONSTRAINT company_info_pkey PRIMARY KEY (symbol);


--
-- Name: ctd ctd_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ctd
    ADD CONSTRAINT ctd_pkey PRIMARY KEY ("time");


--
-- Name: ctg ctg_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ctg
    ADD CONSTRAINT ctg_pkey PRIMARY KEY ("time");


--
-- Name: ctr ctr_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ctr
    ADD CONSTRAINT ctr_pkey PRIMARY KEY ("time");


--
-- Name: cts cts_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.cts
    ADD CONSTRAINT cts_pkey PRIMARY KEY ("time");


--
-- Name: dbc dbc_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dbc
    ADD CONSTRAINT dbc_pkey PRIMARY KEY ("time");


--
-- Name: dcm dcm_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dcm
    ADD CONSTRAINT dcm_pkey PRIMARY KEY ("time");


--
-- Name: dgc dgc_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dgc
    ADD CONSTRAINT dgc_pkey PRIMARY KEY ("time");


--
-- Name: dgw dgw_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dgw
    ADD CONSTRAINT dgw_pkey PRIMARY KEY ("time");


--
-- Name: dig dig_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dig
    ADD CONSTRAINT dig_pkey PRIMARY KEY ("time");


--
-- Name: dpm dpm_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dpm
    ADD CONSTRAINT dpm_pkey PRIMARY KEY ("time");


--
-- Name: dse dse_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dse
    ADD CONSTRAINT dse_pkey PRIMARY KEY ("time");


--
-- Name: dxg dxg_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dxg
    ADD CONSTRAINT dxg_pkey PRIMARY KEY ("time");


--
-- Name: dxs dxs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.dxs
    ADD CONSTRAINT dxs_pkey PRIMARY KEY ("time");


--
-- Name: eib eib_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.eib
    ADD CONSTRAINT eib_pkey PRIMARY KEY ("time");


--
-- Name: evf evf_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.evf
    ADD CONSTRAINT evf_pkey PRIMARY KEY ("time");


--
-- Name: fpt fpt_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.fpt
    ADD CONSTRAINT fpt_pkey PRIMARY KEY ("time");


--
-- Name: frt frt_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.frt
    ADD CONSTRAINT frt_pkey PRIMARY KEY ("time");


--
-- Name: fts fts_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.fts
    ADD CONSTRAINT fts_pkey PRIMARY KEY ("time");


--
-- Name: gas gas_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.gas
    ADD CONSTRAINT gas_pkey PRIMARY KEY ("time");


--
-- Name: gee gee_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.gee
    ADD CONSTRAINT gee_pkey PRIMARY KEY ("time");


--
-- Name: gex gex_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.gex
    ADD CONSTRAINT gex_pkey PRIMARY KEY ("time");


--
-- Name: gmd gmd_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.gmd
    ADD CONSTRAINT gmd_pkey PRIMARY KEY ("time");


--
-- Name: gvr gvr_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.gvr
    ADD CONSTRAINT gvr_pkey PRIMARY KEY ("time");


--
-- Name: hag hag_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hag
    ADD CONSTRAINT hag_pkey PRIMARY KEY ("time");


--
-- Name: hcm hcm_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hcm
    ADD CONSTRAINT hcm_pkey PRIMARY KEY ("time");


--
-- Name: hdb hdb_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hdb
    ADD CONSTRAINT hdb_pkey PRIMARY KEY ("time");


--
-- Name: hdc hdc_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hdc
    ADD CONSTRAINT hdc_pkey PRIMARY KEY ("time");


--
-- Name: hdg hdg_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hdg
    ADD CONSTRAINT hdg_pkey PRIMARY KEY ("time");


--
-- Name: hhv hhv_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hhv
    ADD CONSTRAINT hhv_pkey PRIMARY KEY ("time");


--
-- Name: hpg hpg_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hpg
    ADD CONSTRAINT hpg_pkey PRIMARY KEY ("time");


--
-- Name: hsg hsg_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.hsg
    ADD CONSTRAINT hsg_pkey PRIMARY KEY ("time");


--
-- Name: ht1 ht1_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ht1
    ADD CONSTRAINT ht1_pkey PRIMARY KEY ("time");


--
-- Name: imp imp_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.imp
    ADD CONSTRAINT imp_pkey PRIMARY KEY ("time");


--
-- Name: kbc kbc_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kbc
    ADD CONSTRAINT kbc_pkey PRIMARY KEY ("time");


--
-- Name: kdc kdc_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kdc
    ADD CONSTRAINT kdc_pkey PRIMARY KEY ("time");


--
-- Name: kdh kdh_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kdh
    ADD CONSTRAINT kdh_pkey PRIMARY KEY ("time");


--
-- Name: kos kos_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.kos
    ADD CONSTRAINT kos_pkey PRIMARY KEY ("time");


--
-- Name: lpb lpb_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.lpb
    ADD CONSTRAINT lpb_pkey PRIMARY KEY ("time");


--
-- Name: mbb mbb_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mbb
    ADD CONSTRAINT mbb_pkey PRIMARY KEY ("time");


--
-- Name: msb msb_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.msb
    ADD CONSTRAINT msb_pkey PRIMARY KEY ("time");


--
-- Name: msn msn_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.msn
    ADD CONSTRAINT msn_pkey PRIMARY KEY ("time");


--
-- Name: mwg mwg_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.mwg
    ADD CONSTRAINT mwg_pkey PRIMARY KEY ("time");


--
-- Name: nab nab_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nab
    ADD CONSTRAINT nab_pkey PRIMARY KEY ("time");


--
-- Name: nkg nkg_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nkg
    ADD CONSTRAINT nkg_pkey PRIMARY KEY ("time");


--
-- Name: nlg nlg_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nlg
    ADD CONSTRAINT nlg_pkey PRIMARY KEY ("time");


--
-- Name: nt2 nt2_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nt2
    ADD CONSTRAINT nt2_pkey PRIMARY KEY ("time");


--
-- Name: nvl nvl_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.nvl
    ADD CONSTRAINT nvl_pkey PRIMARY KEY ("time");


--
-- Name: ocb ocb_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ocb
    ADD CONSTRAINT ocb_pkey PRIMARY KEY ("time");


--
-- Name: pan pan_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.pan
    ADD CONSTRAINT pan_pkey PRIMARY KEY ("time");


--
-- Name: pc1 pc1_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.pc1
    ADD CONSTRAINT pc1_pkey PRIMARY KEY ("time");


--
-- Name: pdr pdr_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.pdr
    ADD CONSTRAINT pdr_pkey PRIMARY KEY ("time");


--
-- Name: phr phr_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.phr
    ADD CONSTRAINT phr_pkey PRIMARY KEY ("time");


--
-- Name: plx plx_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.plx
    ADD CONSTRAINT plx_pkey PRIMARY KEY ("time");


--
-- Name: pnj pnj_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.pnj
    ADD CONSTRAINT pnj_pkey PRIMARY KEY ("time");


--
-- Name: pow pow_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.pow
    ADD CONSTRAINT pow_pkey PRIMARY KEY ("time");


--
-- Name: pvd pvd_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.pvd
    ADD CONSTRAINT pvd_pkey PRIMARY KEY ("time");


--
-- Name: pvt pvt_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.pvt
    ADD CONSTRAINT pvt_pkey PRIMARY KEY ("time");


--
-- Name: ree ree_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ree
    ADD CONSTRAINT ree_pkey PRIMARY KEY ("time");


--
-- Name: sab sab_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sab
    ADD CONSTRAINT sab_pkey PRIMARY KEY ("time");


--
-- Name: sbt sbt_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sbt
    ADD CONSTRAINT sbt_pkey PRIMARY KEY ("time");


--
-- Name: scs scs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.scs
    ADD CONSTRAINT scs_pkey PRIMARY KEY ("time");


--
-- Name: shb shb_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.shb
    ADD CONSTRAINT shb_pkey PRIMARY KEY ("time");


--
-- Name: sip sip_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sip
    ADD CONSTRAINT sip_pkey PRIMARY KEY ("time");


--
-- Name: sjs sjs_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.sjs
    ADD CONSTRAINT sjs_pkey PRIMARY KEY ("time");


--
-- Name: ssb ssb_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ssb
    ADD CONSTRAINT ssb_pkey PRIMARY KEY ("time");


--
-- Name: ssi ssi_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.ssi
    ADD CONSTRAINT ssi_pkey PRIMARY KEY ("time");


--
-- Name: stb stb_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.stb
    ADD CONSTRAINT stb_pkey PRIMARY KEY ("time");


--
-- Name: szc szc_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.szc
    ADD CONSTRAINT szc_pkey PRIMARY KEY ("time");


--
-- Name: tch tch_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tch
    ADD CONSTRAINT tch_pkey PRIMARY KEY ("time");


--
-- Name: tpb tpb_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.tpb
    ADD CONSTRAINT tpb_pkey PRIMARY KEY ("time");


--
-- Name: vcb vcb_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vcb
    ADD CONSTRAINT vcb_pkey PRIMARY KEY ("time");


--
-- Name: vcg vcg_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vcg
    ADD CONSTRAINT vcg_pkey PRIMARY KEY ("time");


--
-- Name: vci vci_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vci
    ADD CONSTRAINT vci_pkey PRIMARY KEY ("time");


--
-- Name: vgc vgc_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vgc
    ADD CONSTRAINT vgc_pkey PRIMARY KEY ("time");


--
-- Name: vhc vhc_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vhc
    ADD CONSTRAINT vhc_pkey PRIMARY KEY ("time");


--
-- Name: vhm vhm_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vhm
    ADD CONSTRAINT vhm_pkey PRIMARY KEY ("time");


--
-- Name: vib vib_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vib
    ADD CONSTRAINT vib_pkey PRIMARY KEY ("time");


--
-- Name: vic vic_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vic
    ADD CONSTRAINT vic_pkey PRIMARY KEY ("time");


--
-- Name: vix vix_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vix
    ADD CONSTRAINT vix_pkey PRIMARY KEY ("time");


--
-- Name: vjc vjc_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vjc
    ADD CONSTRAINT vjc_pkey PRIMARY KEY ("time");


--
-- Name: vnd vnd_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vnd
    ADD CONSTRAINT vnd_pkey PRIMARY KEY ("time");


--
-- Name: vnm vnm_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vnm
    ADD CONSTRAINT vnm_pkey PRIMARY KEY ("time");


--
-- Name: vpb vpb_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vpb
    ADD CONSTRAINT vpb_pkey PRIMARY KEY ("time");


--
-- Name: vpi vpi_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vpi
    ADD CONSTRAINT vpi_pkey PRIMARY KEY ("time");


--
-- Name: vpl vpl_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vpl
    ADD CONSTRAINT vpl_pkey PRIMARY KEY ("time");


--
-- Name: vre vre_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vre
    ADD CONSTRAINT vre_pkey PRIMARY KEY ("time");


--
-- Name: vsc vsc_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vsc
    ADD CONSTRAINT vsc_pkey PRIMARY KEY ("time");


--
-- Name: vtp vtp_pkey; Type: CONSTRAINT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY public.vtp
    ADD CONSTRAINT vtp_pkey PRIMARY KEY ("time");


--
-- PostgreSQL database dump complete
--

\unrestrict wKqHBNYrdkeaOFcuhjBlO5MJsdl2RvwotuSk1WuCUtFw7zru0wMzyeZA18mZ1mV

