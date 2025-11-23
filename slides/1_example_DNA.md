---
zoom: 0.95
---

# Example. Cancer Cell Classification

<div class="grid grid-cols-[3fr_3fr] gap-5">
<div>

* Human cells contain [DNA](https://en.wikipedia.org/wiki/DNA)
	* DNA is a **double-helix** sequence of **nucleotides**: A, C, G, T
* **Genes** are patterns (subsequences) in DNA
	* Genes determine biological traits
		* Eye color, height,<br> **risk for diseases**, ...
* A human [chromosome](https://en.wikipedia.org/wiki/Chromosome)<br> is a tightly packed DNA
* Human have 23 chromosome pairs
	* Each has 500M nucleotides and **thousands of genes**
</div>
<div>
	<br>
  <figure>
    <img src="/DNA_Structure+Key+Labelled.pn_NoBB.png" style="width: 400px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>Image source:
	  <a href="https://en.wikipedia.org/wiki/DNA">https://en.wikipedia.org/wiki/DNA</a>
    </figcaption>
  </figure>
</div>
</div>

---
zoom: 0.93
---

# Example. Cancer Cell Classification

<div class="grid grid-cols-[7fr_2fr] gap-5">
<div>
  <figure>
    <img src="/Chromosome_DNA_Gene.svg.png" style="width: 190px !important">
  </figure>

* Gene expression values from microarray experiment
	* Genes are sprayed with substance that expresses them in front of high resolution camera, which reads light intensities into a matrix
* Hi-Dim data, $p >> n$:
	* $n$ cells types and $p$ genes
	* $n$ patients and $p$ genes
	* $n$ patients and $p$ cells
* We might want find relations between **genes** and **diseases** 
	* E.g. whether a cell is likely to have cancer
</div>
<div>
  <figure>
    <img src="/genes.png" style="width: 200px !important">
    <figcaption style="color:#b3b3b3ff; font-size: 11px;"><br>Images sources:
	  <a href="https://www.pnas.org/doi/10.1073/pnas.0308531101">https://www.pnas.org/doi/10.1073/pnas.0308531101</a>
	  <br><a href="https://en.wikipedia.org/wiki/Gene">https://en.wikipedia.org/wiki/Gene</a>
    </figcaption>
  </figure>
</div>
</div>