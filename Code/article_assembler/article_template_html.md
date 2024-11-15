<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{{car_brand}} {{car_model}} - The Future of Driving</title>
    <style>
        /* General page styling */
        body {
            font-family: Arial, sans-serif;
            color: #EAEAEA;
            background-color: #121212;
            padding: 20px;
            line-height: 1.6;
        }
        h1 {
            font-size: 2.5em;
            color: #FFD700;
            text-align: center;
            font-family: 'Montserrat', sans-serif;
            font-weight: bold;
        }
        h2 {
            font-size: 1.8em;
            color: #87CEEB;
            border-bottom: 2px solid #87CEEB;
            padding-bottom: 5px;
            font-family: 'Roboto', sans-serif;
        }
        p {
            font-size: 1.1em;
            color: #D3D3D3;
            font-family: 'Open Sans', sans-serif;
            margin: 15px 0;
        }
        figure.left {
            float: left;
            width: 45%;
            margin: 20px 5% 20px 0;
            text-align: center;
        }
        figure.right {
            float: right;
            width: 45%;
            margin: 20px 0 20px 5%;
            text-align: center;
        }
        figcaption {
            font-size: 0.9em;
            color: #A9A9A9;
            margin-top: 5px;
        }
        img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
        }
        .button {
            display: inline-block;
            padding: 10px 20px;
            font-size: 1.1em;
            color: #121212;
            background-color: #FFD700;
            border-radius: 5px;
            text-decoration: none;
            margin-top: 20px;
        }
        .button:hover {
            background-color: #FFB800;
        }
    </style>
</head>
<body>

<h1>{{car_brand}} {{car_model}} - The Future of Driving</h1>

<h2>Introduction</h2>
<p>{{paragraph_1}}</p>

<h2>Design and Style</h2>
<p>{{paragraph_2}}</p>

<figure class="left">
    <img src="figure_1" alt="Exterior View">
    <figcaption>{{caption_1}}</figcaption>
</figure>

<h2>Performance and Efficiency</h2>
<p>{{paragraph_3}}</p>

<figure class="right">
    <img src="figure_2" alt="Performance">
    <figcaption>{{caption_2}}</figcaption>
</figure>

<h2>Interior and Comfort</h2>
<p>{{paragraph_4}}</p>

<figure class="left">
    <img src="figure_3" alt="Interior">
    <figcaption>{{caption_3}}</figcaption>
</figure>

<h2>Safety and Technology</h2>
<p>{{paragraph_5}}</p>

<figure class="right">
    <img src="figure_4" alt="Safety Features">
    <figcaption>{{caption_4}}</figcaption>
</figure>

<h2>Conclusion</h2>
<p>{{paragraph_6}}</p>

<p style="text-align:center;">
    <a href="#" class="button">Book a Test Drive Today!</a>
</p>

</body>
</html>
